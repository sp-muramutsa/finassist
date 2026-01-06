from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkMeta(BaseModel):
    source: str = Field(..., description="Filename of the source PDF")
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    chunk_id: int = Field(..., ge=0, description="Chunk identifier within the page")


class Chunk(BaseModel):
    meta: ChunkMeta
    text: str = Field(..., min_length=1)


class SearchHit(BaseModel):
    chunk: Chunk
    score: float


class SearchResponse(BaseModel):
    results: List[SearchHit]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, description="User query to answer")
    top_k: int = Field(4, ge=1, le=20, description="Number of retrieved chunks")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="LLM sampling temperature")
    model: Optional[str] = Field(None, description="Override model name if needed")


class ChatResponse(BaseModel):
    answer: str
    citations: List[Chunk]
    used_model: str
