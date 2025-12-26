# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_engine import answer_question

app = FastAPI(title="RAG Assistant")  # убрал "Pro"


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return answer_question(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
