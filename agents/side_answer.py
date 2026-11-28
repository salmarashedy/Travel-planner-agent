import logging
import re

from pydantic import BaseModel, ValidationError

from agents.extract import StructuredLLM
from core.exceptions import LLMClientError

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a travel assistant helping someone plan a trip. Answer their question in one "
    "or two short sentences of plain English. Never mention prices, costs or amounts of "
    "money; live prices are checked later. Don't offer to look anything up or do anything "
    "else. If you're not sure, say so briefly."
)
SAFE_FALLBACK = "I'll check live prices and details when I build your plan."
MAX_CHARS = 400
MAX_TOKENS = 200  # without a cap the model can ramble until the request times out

PRICE = re.compile(
    r"[$€£]|\b(eur|usd|egp|gbp|euros?|dollars?|pounds?|bucks)\b",
    re.IGNORECASE,
)


class SideAnswer(BaseModel):
    answer: str


# asking for JSON stops the thinking model from writing out its reasoning
SCHEMA = SideAnswer.model_json_schema()


def first_sentences(text: str, count: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = sentences[:count]
    # drop a half sentence left by the token limit, unless it's all we have
    if len(kept) > 1 and not kept[-1].endswith((".", "!", "?")):
        kept = kept[:-1]
    return " ".join(kept)


def safe(answer: str) -> str | None:
    answer = first_sentences(answer)
    if not answer:
        return None
    if PRICE.search(answer) or len(answer) > MAX_CHARS:
        return SAFE_FALLBACK
    return answer


def read_answer(raw: str) -> str | None:
    try:
        return SideAnswer.model_validate_json(raw).answer
    except ValidationError:
        # cut off by the token limit: keep what was written so far
        match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
        return match.group(1).replace('\\"', '"') if match else None


async def answer_side_question(llm: StructuredLLM, question: str) -> str | None:
    # the question goes in as a user message, never into the system prompt
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    try:
        raw = await llm.chat_structured(messages, SCHEMA, max_tokens=MAX_TOKENS)
    except LLMClientError as exc:
        logger.warning("side answer failed: %s", exc)
        return None
    answer = read_answer(raw)
    return safe(answer) if answer else None
