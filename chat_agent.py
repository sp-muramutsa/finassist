"""Chat agent that orchestrates MCP search + OpenAI with Pydantic models."""

from __future__ import annotations

import os
import requests
import openai
from typing import List

from models import ChatRequest, ChatResponse, SearchHit, Chunk

openai.api_key = os.environ.get("OPENAI_API_KEY")
MCP_URL = os.environ.get("MCP_URL", "http://localhost:8001")
MCP_KEY = os.environ.get("MCP_API_KEY", "devkey")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def call_mcp_search(query: str, top_k: int) -> List[SearchHit]:
    resp = requests.post(
        f"{MCP_URL}/mcp/search",
        headers={"Authorization": f"Bearer {MCP_KEY}"},
        json={"query": query, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    return [SearchHit(**item) for item in payload.get("results", [])]


def build_system_prompt() -> str:
    return (
        "You are FinAssist, a financial-document assistant. "
        "Use the provided evidence to craft concise answers with citations (source + page). "
        "If evidence is weak or missing, state assumptions and suggest follow-up data to fetch."
    )


def respond(req: ChatRequest) -> ChatResponse:
    try:
        hits = call_mcp_search(req.question, top_k=req.top_k)
    except Exception:
        hits = []

    context_snippets = []
    for h in hits:
        context_snippets.append(
            f"Source: {h.chunk.meta.source} page {h.chunk.meta.page} score:{h.score}\n{h.chunk.text[:800]}"
        )

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": req.question},
    ]
    if context_snippets:
        messages.append(
            {
                "role": "assistant",
                "content": "Use the following evidence to answer and cite sources:\n"
                + "\n---\n".join(context_snippets),
            }
        )

    model_name = req.model or DEFAULT_MODEL
    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=req.temperature,
        max_tokens=600,
    )
    answer = resp["choices"][0]["message"]["content"]
    citations: List[Chunk] = [h.chunk for h in hits]

    return ChatResponse(answer=answer, citations=citations, used_model=model_name)


if __name__ == "__main__":
    demo_request = ChatRequest(
        question="Summarize the revenue recognition policy referenced in the attached 2023 financial report.",
        top_k=4,
        temperature=0.0,
    )
    print(respond(demo_request))
