# Travel Agent Backend Foundation

A clean FastAPI foundation for a future AI-powered travel-planning agent.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Documentation is at `/docs` and health is at `/health`.

## Test

```bash
pytest
```

The chat endpoint currently validates the future contract and returns HTTP 501. It does not require Ollama to be running and does not yet implement travel-agent or LangGraph behavior.
