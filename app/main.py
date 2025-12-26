from fastapi import FastAPI

app = FastAPI(title="RAG Assistant Pro")

@app.get("/health")
def health():
    return {"status": "ok"}
