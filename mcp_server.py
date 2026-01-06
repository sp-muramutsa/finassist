"""MCP-style search server powered by FAISS and OpenAI embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
import openai
from fastapi import APIRouter, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from models import Chunk, ChunkMeta, SearchHit, SearchResponse

MCP_API_KEY = os.environ.get("MCP_API_KEY", "devkey")
INDEX_PATH = Path(os.environ.get("INDEX_PATH", "faiss.index"))
META_PATH = Path(os.environ.get("META_PATH", "meta.json"))
openai.api_key = os.environ.get("OPENAI_API_KEY")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)


def embed_query(q: str):
    resp = openai.Embedding.create(model="text-embedding-3-small", input=[q])
    return np.array(resp["data"][0]["embedding"]).astype("float32")


class VectorStore:
    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metas: List[dict] = []

    def load(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise RuntimeError(
                f"Vector store not initialized. Expected {self.index_path} and {self.meta_path}. Run ingest.py first."
            )
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r") as f:
            self.metas = json.load(f)

    def ensure_loaded(self):
        if self.index is None or not self.metas:
            self.load()

    def search(self, query: str, top_k: int) -> List[SearchHit]:
        self.ensure_loaded()
        q_emb = embed_query(query)
        D, I = self.index.search(np.array([q_emb]), top_k)
        results: List[SearchHit] = []
        for idx, score in zip(I[0], D[0]):
            meta_record = self.metas[idx]
            chunk = Chunk(
                meta=ChunkMeta(**meta_record["meta"]),
                text=meta_record["text"],
            )
            results.append(SearchHit(chunk=chunk, score=float(score)))
        return results

    def fetch(self, source: str, page: int, chunk_id: int) -> Chunk:
        self.ensure_loaded()
        for meta_record in self.metas:
            meta = meta_record["meta"]
            if meta["source"] == source and meta["page"] == page and meta["chunk_id"] == chunk_id:
                return Chunk(meta=ChunkMeta(**meta), text=meta_record["text"])
        raise KeyError("Chunk not found")


router = APIRouter(prefix="/mcp", tags=["mcp"])
store = VectorStore(INDEX_PATH, META_PATH)


def authorize(header: str | None):
    if header != f"Bearer {MCP_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/search", response_model=SearchResponse)
def search_mcp(req: SearchRequest, authorization: str = Header(None)):
    authorize(authorization)
    hits = store.search(req.query, req.top_k)
    return SearchResponse(results=hits)


@router.post("/fetch", response_model=Chunk)
def fetch_mcp(item: ChunkMeta, authorization: str = Header(None)):
    authorize(authorization)
    try:
        return store.fetch(item.source, item.page, item.chunk_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Not found")


def create_app() -> FastAPI:
    app = FastAPI(title="FinAssist MCP")

    @app.on_event("startup")
    async def _load_store():
        try:
            store.load()
        except Exception as exc:
            # Defer failure to first request so app can start even before indexing.
            print(f"[warn] MCP store not preloaded: {exc}")

    app.include_router(router)
    return app


mcp_app = create_app()
app = mcp_app  # for `uvicorn mcp_server:app`
