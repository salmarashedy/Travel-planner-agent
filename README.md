# Travel Agent Backend

A FastAPI backend for an AI travel-planning agent. Feature 1 is in place: the
bot chats with the traveller, fills in a trip form (country, dates, budget,
people...), and asks for a final "yes" before handing the trip on.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

The API is at `http://127.0.0.1:8000`, with docs at `/docs` and health at `/health`.

The chat needs [Ollama](https://ollama.com) running with the model from `.env`:

```bash
ollama pull qwen3:4b-thinking-2507-q4_K_M
```

The app and `/health` start fine without Ollama; only chat replies need it.

## Chat API

`POST /chat`

```json
{ "session_id": "session_12345", "message": "7 days in Italy, 1500 euros", "user_id": "user_123" }
```

```json
{
  "reply": "What day would you like the trip to start? How many people are travelling, counting children?",
  "trip_spec": { "country": "Italy", "num_days": 7, "budget_amount": 1500.0, "budget_currency": "EUR", "...": "..." },
  "status": "collecting"
}
```

`status` is `collecting` while the form is being filled, `confirmed` once the
traveller says yes to the summary, and `error` (with `trip_spec: null`) if
something unexpected went wrong. Each `session_id` is its own chat.

## How a message is handled

```
extract --(model failed)--> fail
   |
 merge --> decide --> reply
```

- `agents/extract.py` - Qwen3 reads the message and reports what it found (the only AI step).
- `agents/merge.py` + `agents/grounding.py` - Python checks each value is really in the message and valid, then saves it.
- `agents/decide.py` - Python picks the next move: ask, clarify, show the summary, or confirm.
- `agents/reply.py` - the reply, from fixed templates.
- `agents/graph.py` - LangGraph wiring; chats are saved per session (in memory for now).

The prompt is in `agents/prompts/` so it can be improved without touching code.

## Test

```bash
python -m pytest
```

Tests use a fake model, so they need no Ollama, GPU, or network. They also run
on GitHub Actions on every push.

Real-model checks are kept apart and skipped by default. With Ollama running:

```bash
python -m pytest -m llm                                  # golden-set checks with the real model
python -m scripts.chat                                   # chat in the terminal
python -m scripts.try_extract "10 days in Greece"        # see what the model finds
python -m scripts.compare_models qwen3:4b-thinking-2507-q4_K_M --merge        # score it on the dev messages
python -m scripts.compare_models qwen3:4b-thinking-2507-q4_K_M --merge --cases heldout --log
mlflow ui --backend-store-uri sqlite:///mlflow.db        # compare saved runs
```

The golden messages are in `tests/golden/`: `extract_dev.json` is for working
on the prompt, `extract_heldout.json` is only for measuring, never for tuning.
