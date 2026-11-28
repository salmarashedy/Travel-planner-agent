# Score models on the golden cases:
#   python -m scripts.compare_models qwen3:4b-thinking-2507-q4_K_M gemma3:4b --show-errors
#   python -m scripts.compare_models qwen3:4b-thinking-2507-q4_K_M --merge --cases heldout --log

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import get_args

from agents.extract import PROMPT_VERSION, extract
from agents.grounding import tidy_extraction
from agents.merge import merge
from core.config import get_settings
from schemas.country import normalize_country
from schemas.trip import Extraction, FieldName, TripSpec
from services.llm_client import OllamaClient

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden"


def load_cases(name: str) -> tuple[list[dict], date]:
    data = json.loads((GOLDEN_DIR / f"extract_{name}.json").read_text("utf-8"))
    return data["cases"], date.fromisoformat(data["today"])


@dataclass
class Score:
    cases: int = 0
    type_ok: int = 0
    found: int = 0
    expected: int = 0
    invented: int = 0
    failed: int = 0
    seconds: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def found_rate(self) -> float:
        return self.found / self.expected if self.expected else 1.0

    @property
    def type_rate(self) -> float:
        return self.type_ok / self.cases if self.cases else 1.0

    @property
    def p95_seconds(self) -> float:
        ordered = sorted(self.seconds)
        return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


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


async def run_model(model: str, cases: list[dict], today: date, with_merge: bool) -> Score:
    settings = get_settings().model_copy(update={"ollama_model": model})
    llm = OllamaClient(settings)
    score = Score(cases=len(cases))

    # warm up so loading the model isn't timed
    await extract(llm, "hi", TripSpec(), today)

    for case in cases:
        start = time.perf_counter()
        result = await extract(
            llm, case["message"], TripSpec(), today, last_question=case.get("last_question")
        )
        score.seconds.append(time.perf_counter() - start)
        if with_merge and result is not None:
            result = after_merge(tidy_extraction(result, case["message"]), case["message"], today)
            case = {**case, "updates": {k: v for k, v in case.get("updates", {}).items() if k != "end_date"}}
        score_case(case, result, score)
    return score


def after_merge(result: Extraction, message: str, today: date) -> Extraction:
    merged = merge(TripSpec(), result, message, today)
    kept = merged.spec.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    interests = kept.pop("interests", [])
    kept.pop("from_memory", None)
    sentences = " ".join(merged.conflicts)
    conflicts = [
        {"field": name, "options": ["?", "?"]}
        for name in get_args(FieldName)
        if f"For {name.replace('_', ' ')} you said" in sentences
    ]
    return Extraction(
        message_type=result.message_type,
        updates=kept,
        interests_add=interests,
        unclear=result.unclear,
        conflicts=conflicts,
        side_question=merged.side_question,
    )


def log_run(model: str, score: Score, cases_name: str, with_merge: bool) -> None:
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("feature1-extraction")
    with mlflow.start_run(run_name=f"{model} {cases_name}{' +merge' if with_merge else ''}"):
        mlflow.log_params({
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "cases": cases_name,
            "merge": with_merge,
            "num_cases": score.cases,
        })
        mlflow.log_metrics({
            "type_accuracy": score.type_rate,
            "found_rate": score.found_rate,
            "invented": score.invented,
            "failed": score.failed,
            "latency_p50_s": statistics.median(score.seconds),
            "latency_p95_s": score.p95_seconds,
        })


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--cases", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--merge", action="store_true", help="score what reaches the form after step 5")
    parser.add_argument("--log", action="store_true", help="save each run to MLflow")
    args = parser.parse_args()

    cases, today = load_cases(args.cases)
    n = len(cases)

    rows = []
    for model in args.models:
        print(f"running {model} on {n} {args.cases} cases...")
        score = await run_model(model, cases, today, args.merge)
        rows.append((model, score))
        if args.show_errors:
            for line in score.errors:
                print("   ", line)
        if args.log:
            log_run(model, score, args.cases, args.merge)

    print()
    print(f"{'model':<34}{'type ok':>9}{'found':>9}{'invented':>10}{'failed':>8}{'p50 s':>8}{'p95 s':>8}")
    for model, s in rows:
        print(
            f"{model:<34}{s.type_ok:>4}/{n:<4}{s.found:>4}/{s.expected:<4}"
            f"{s.invented:>10}{s.failed:>8}{statistics.median(s.seconds):>8.1f}{s.p95_seconds:>8.1f}"
        )
    if args.log:
        print(f"\nsaved to MLflow. To view: mlflow ui --backend-store-uri {get_settings().mlflow_tracking_uri}")


if __name__ == "__main__":
    asyncio.run(main())
