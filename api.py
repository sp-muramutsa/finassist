"""FastAPI application entrypoint for the FinAssist service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chat_agent import respond
from mcp_server import router as mcp_router
from models import ChatRequest, ChatResponse

app = FastAPI(title="FinAssist API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return respond(req)


# Expose MCP routes under /mcp
app.include_router(mcp_router)
