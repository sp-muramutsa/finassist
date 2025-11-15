# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from chat_agent import respond

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    ans = respond(req.question)
    return {"answer": ans}
