# Real-model checks, skipped unless you run: python -m pytest -m llm
# The limits sit a bit below what qwen3:4b scored, so they catch it getting worse.

import httpx
import pytest

from core.config import get_settings
from scripts.compare_models import load_cases, run_model

pytestmark = pytest.mark.llm

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
