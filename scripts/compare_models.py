"""Score extraction on the dev cases for one or more models.

    python -m scripts.compare_models qwen3:4b llama3.2:3b
    python -m scripts.compare_models qwen3:4b --show-errors
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from agents.extract import extract
from core.config import get_settings
from schemas.country import normalize_country
from schemas.trip import Extraction, TripSpec
from services.llm_client import OllamaClient

CASES_FILE = Path(__file__).resolve().parent.parent / "tests" / "golden" / "extract_dev.json"


@dataclass
class Score:
    type_ok: int = 0
    found: int = 0
    expected: int = 0
    invented: int = 0
    failed: int = 0
    seconds: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def same(name: str, got, want) -> bool:
    if got is None:
        return False
    if name == "country":
        try:
            return normalize_country(str(got)) == normalize_country(str(want))
        except ValueError:
            return False
    if name == "budget_currency":
        try:
            return TripSpec(budget_currency=got).budget_currency == want
        except ValueError:
            return False
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and float(got) == float(want)
    return str(got).strip().lower() == str(want).strip().lower()


def score_case(case: dict, result: Extraction | None, score: Score) -> None:
    want_updates = case.get("updates", {})
    want_interests = set(case.get("interests_add", []))
    want_unclear = set(case.get("unclear", []))
    want_conflicts = set(case.get("conflicts", []))
    want_side = case.get("side_question", False)
    checks = len(want_updates) + len(want_interests) + len(want_unclear) + len(want_conflicts) + want_side
    score.expected += checks

    if result is None:
        score.failed += 1
        score.errors.append(f"{case['message']!r}: no usable answer")
        return

    problems = []
    if result.message_type == case["type"]:
        score.type_ok += 1
    else:
        problems.append(f"type {result.message_type} (want {case['type']})")

    got = result.updates.model_dump(exclude_none=True)
    for name, want in want_updates.items():
        if same(name, got.get(name), want):
            score.found += 1
        else:
            problems.append(f"{name}={got.get(name)!r} (want {want!r})")

    for name, value in got.items():
        if name not in want_updates and name not in case.get("allow", []):
            score.invented += 1
            problems.append(f"invented {name}={value!r}")

    got_interests = set(result.interests_add)
    score.found += len(got_interests & want_interests)
    score.invented += len(got_interests - want_interests)
    if got_interests != want_interests:
        problems.append(f"interests {sorted(got_interests)} (want {sorted(want_interests)})")
    if result.interests_remove:
        score.invented += len(result.interests_remove)
        problems.append(f"invented interests_remove {result.interests_remove}")

    score.found += len(set(result.unclear) & want_unclear)
    if want_unclear - set(result.unclear):
        problems.append(f"unclear {result.unclear} (want {sorted(want_unclear)})")

    got_conflicts = {c.field for c in result.conflicts}
    score.found += len(got_conflicts & want_conflicts)
    if got_conflicts != want_conflicts:
        problems.append(f"conflicts {sorted(got_conflicts)} (want {sorted(want_conflicts)})")

    if want_side and result.side_question:
        score.found += 1
    elif want_side:
        problems.append("missed side_question")
    elif result.side_question:
        score.invented += 1
        problems.append(f"invented side_question {result.side_question!r}")

    if problems:
        score.errors.append(f"{case['message']!r}: " + "; ".join(problems))


async def run_model(model: str, cases: list[dict], today: date) -> Score:
    settings = get_settings().model_copy(update={"ollama_model": model})
    llm = OllamaClient(settings)
    score = Score()

    # Load the model first so the first case isn't timed with the load.
    await extract(llm, "hi", TripSpec(), today)

    for case in cases:
        start = time.perf_counter()
        result = await extract(
            llm, case["message"], TripSpec(), today, last_question=case.get("last_question")
        )
        score.seconds.append(time.perf_counter() - start)
        score_case(case, result, score)
    return score


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    data = json.loads(CASES_FILE.read_text("utf-8"))
    cases, today = data["cases"], date.fromisoformat(data["today"])
    n = len(cases)

    rows = []
    for model in args.models:
        print(f"running {model} on {n} cases...")
        score = await run_model(model, cases, today)
        rows.append((model, score))
        if args.show_errors:
            for line in score.errors:
                print("   ", line)

    print()
    print(f"{'model':<34}{'type ok':>9}{'found':>9}{'invented':>10}{'failed':>8}{'avg s':>8}")
    for model, s in rows:
        print(
            f"{model:<34}{s.type_ok:>4}/{n:<4}{s.found:>4}/{s.expected:<4}"
            f"{s.invented:>10}{s.failed:>8}{statistics.mean(s.seconds):>8.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
