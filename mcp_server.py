# mcp_server.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import json, faiss, numpy as np, os
import openai

app = FastAPI()
MCP_API_KEY = os.environ.get("MCP_API_KEY", "devkey")
INDEX_PATH = "faiss.index"
META_PATH = "meta.json"
openai.api_key = os.environ.get("OPENAI_API_KEY")

# load FAISS + meta
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r") as f: metas = json.load(f)
dim = index.d

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

def embed_query(q: str):
    resp = openai.Embedding.create(model="text-embedding-3-small", input=[q])
    return np.array(resp["data"][0]["embedding"]).astype("float32")

@app.post("/mcp/search")
def search_mcp(req: SearchRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {MCP_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized to use MCP")
    q_emb = embed_query(req.query)
    D, I = index.search(np.array([q_emb]), req.top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        meta = metas[idx]
        results.append({"meta": meta, "score": float(score)})
    return {"results": results}

@app.post("/mcp/fetch")
def fetch_mcp(item: dict, authorization: str = Header(None)):
    # item: {"source": "file.pdf", "page": 2, "chunk_id": 0}
    if authorization != f"Bearer {MCP_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    # find first meta match (simple)
    for m, meta in enumerate(metas):
        if meta["meta"]["source"] == item["source"] and meta["meta"]["page"] == item["page"] and meta["meta"]["chunk_id"] == item["chunk_id"]:
            return {"text": meta["text"], "meta": meta}
    raise HTTPException(status_code=404, detail="Not found")
