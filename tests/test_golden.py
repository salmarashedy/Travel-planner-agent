# Real-model checks, skipped unless you run: python -m pytest -m llm
# The limits sit a bit below what qwen3:4b scored, so they catch it getting worse.

from datetime import date

import httpx
import pytest

from agents.extract import build_messages, extract
from core.config import get_settings
from schemas.trip import TripSpec
from scripts.compare_models import load_cases, run_model
from services.llm_client import OllamaClient

pytestmark = pytest.mark.llm

TODAY = date(2026, 11, 21)

LIMITS = {
    # (min type accuracy, min share of values found)
    "dev": (0.90, 0.70),
    "heldout": (0.85, 0.70),
}


@pytest.fixture(scope="module", autouse=True)
def needs_ollama():
    settings = get_settings()
    try:
        tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3).json()
    except httpx.HTTPError:
        pytest.skip("Ollama is not running")
    if settings.ollama_model not in {m["name"] for m in tags.get("models", [])}:
        pytest.skip(f"{settings.ollama_model} is not pulled")


@pytest.mark.parametrize("cases_name", ["dev", "heldout"])
async def test_model_on_golden_cases(cases_name):
    cases, today = load_cases(cases_name)
    score = await run_model(get_settings().ollama_model, cases, today, with_merge=True)
    report = "\n".join(score.errors)
    min_type, min_found = LIMITS[cases_name]

    assert score.invented == 0, f"made-up values reached the form:\n{report}"
    assert score.failed == 0, f"no usable answer for some messages:\n{report}"
    assert score.type_rate >= min_type, f"type accuracy {score.type_rate:.0%}:\n{report}"
    assert score.found_rate >= min_found, f"found {score.found_rate:.0%} of values:\n{report}"


async def test_same_message_gives_the_same_answer():
    llm = OllamaClient(get_settings())
    message = "me, my wife and our two kids, around 2k each"
    answers = {
        (await extract(llm, message, TripSpec(), TODAY)).model_dump_json() for _ in range(5)
    }
    assert len(answers) == 1, answers


async def test_prompt_fits_in_the_context_window():
    settings = get_settings()
    full = TripSpec(country="Italy", region="Rome", start_date=date(2027, 6, 10), num_days=7,
                    budget_amount=1500, budget_currency="EUR", party_size=4, hotel_stars=4,
                    min_review_score=8, interests=["history", "food", "nature"])
    history = [{"role": "user", "content": "x " * 500}, {"role": "assistant", "content": "y " * 150}]
    messages = build_messages("z " * 500, full, TODAY, last_question="Is this right?", history=history)

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{settings.ollama_base_url}/api/chat", json={
            "model": settings.ollama_model, "messages": messages, "stream": False, "think": False,
            "options": {"num_ctx": 4 * settings.ollama_num_ctx, "num_predict": 1},
        })
    used = r.json()["prompt_eval_count"]
    # leave at least half the window for the answer; it was about 40% when written
    assert used < 0.5 * settings.ollama_num_ctx, f"{used} prompt tokens"
