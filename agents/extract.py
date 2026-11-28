import json
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from string import Template
from typing import Any, Protocol

from pydantic import ValidationError

from core.exceptions import LLMClientError
from schemas.trip import Extraction, TripSpec

logger = logging.getLogger(__name__)

PROMPT_VERSION = "extract_v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT = Template((PROMPTS_DIR / f"{PROMPT_VERSION}.txt").read_text("utf-8"))
EXAMPLES = json.loads((PROMPTS_DIR / f"{PROMPT_VERSION}_examples.json").read_text("utf-8"))
SCHEMA = Extraction.model_json_schema()
HISTORY_TURNS = 2


class StructuredLLM(Protocol):
    async def chat_structured(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int | None = None
    ) -> str: ...


def next_occurrence(month: int, day: int, today: date) -> date:
    candidate = date(today.year, month, day)
    return candidate if candidate >= today else candidate.replace(year=today.year + 1)


def year_rule(today: date) -> str:
    # small models get the year wrong, so give them real dates
    jan, dec = next_occurrence(1, 15, today), next_occurrence(12, 20, today)
    return (
        f"A date without a year means its next time: on or after {today:%d %B} is "
        f"{today.year}, before {today:%d %B} is {today.year + 1}. "
        f'For example "January 15" is {jan.isoformat()} and "December 20" is {dec.isoformat()}.'
    )


def example_turns(today: date) -> list[dict[str, str]]:
    # qwen3:4b follows examples much better as chat turns than in the prompt
    dates = {
        "{jan_15}": next_occurrence(1, 15, today).isoformat(),
        "{jan_19}": next_occurrence(1, 19, today).isoformat(),
    }
    turns = []
    for example in EXAMPLES:
        answer = json.dumps(example["answer"])
        for placeholder, value in dates.items():
            answer = answer.replace(placeholder, value)
        turns.append({"role": "user", "content": example["message"]})
        turns.append({"role": "assistant", "content": answer})
    return turns


def build_messages(
    message: str,
    spec: TripSpec,
    today: date,
    last_question: str | None = None,
    history: Sequence[dict[str, str]] = (),
) -> list[dict[str, str]]:
    known = spec.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_TURNS:])

    system = PROMPT.substitute(
        today=today.strftime("%A %d %B %Y"),
        year_rule=year_rule(today),
        trip_spec=json.dumps(known) if known else "nothing yet",
        last_question=last_question or "none",
        history=recent or "none",
        schema=json.dumps(SCHEMA),
    )
    return [
        {"role": "system", "content": system},
        *example_turns(today),
        {"role": "user", "content": message},
    ]


async def extract(
    llm: StructuredLLM,
    message: str,
    spec: TripSpec,
    today: date,
    last_question: str | None = None,
    history: Sequence[dict[str, str]] = (),
) -> Extraction | None:
    messages = build_messages(message, spec, today, last_question, history)

    for attempt in (1, 2):
        try:
            raw = await llm.chat_structured(messages, SCHEMA)
        except LLMClientError as exc:
            # no point retrying if Ollama is down
            logger.warning("extraction call failed: %s", exc)
            return None

        try:
            return Extraction.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("invalid extraction (attempt %d): %s", attempt, exc)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"That JSON was not valid:\n{exc}\nReply again with corrected JSON only.",
                },
            ]

    return None
