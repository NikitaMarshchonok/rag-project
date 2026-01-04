# RAG Assistant Pro

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
uvicorn rag_assistant.app.main:app --reload --port 8000
```

## Check
http://127.0.0.1:8000/health


## Author

Nikita M.

