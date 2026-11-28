import pytest

from agents.side_answer import SAFE_FALLBACK, SYSTEM, answer_side_question, safe
from core.exceptions import LLMClientError
from tests.fakes import FakeLLM


@pytest.mark.parametrize(
    "answer",
    [
        "Rooms cost about 120 euros a night.",
        "Expect to pay around $90.",
        "Dinner is roughly £30.",
        "Hotels start at 100 EUR.",
        "A budget of 50 dollars a day is enough.",
        "Most meals are under 200 EGP.",
    ],
)
def test_answers_with_prices_are_blocked(answer):
    assert safe(answer) == SAFE_FALLBACK


def test_only_the_first_two_sentences_are_kept():
    # from a real chat: qwen3:4b gave five sentences and offered to check the weather
    answer = (
        "Rome is generally nice in winter. The weather is mild. However, it's a bit chilly. "
        "If you're not sure, I can check the current weather for you later."
    )
    assert safe(answer) == "Rome is generally nice in winter. The weather is mild."


def test_a_price_in_a_later_sentence_does_not_matter():
    assert safe("It's mild. It's quiet. Hotels are about 80 euros.") == "It's mild. It's quiet."


def test_very_long_sentences_are_blocked():
    assert safe("Rome is lovely " * 40 + ".") == SAFE_FALLBACK


def test_normal_answer_passes():
    assert safe("  Rome is mild in winter and much quieter than in summer.  ") == (
        "Rome is mild in winter and much quieter than in summer."
    )


def test_words_like_expensive_are_fine():
    assert safe("Rome can be expensive in summer, but winter is cheaper.") != SAFE_FALLBACK


def test_empty_answer_gives_nothing():
    assert safe("   ") is None


async def test_question_goes_in_as_a_user_message():
    llm = FakeLLM({})
    await answer_side_question(llm, "Is Rome nice in winter?")
    system, user = llm.side_calls[0]
    assert system == {"role": "system", "content": SYSTEM}
    assert user == {"role": "user", "content": "Is Rome nice in winter?"}


async def test_model_error_gives_nothing():
    llm = FakeLLM({}, side_answers={"Is Rome nice in winter?": LLMClientError("down")})
    assert await answer_side_question(llm, "Is Rome nice in winter?") is None


class Raw:
    def __init__(self, raw):
        self.raw = raw
        self.max_tokens = None

    async def chat_structured(self, messages, schema, max_tokens=None):
        self.max_tokens = max_tokens
        return self.raw


async def test_broken_json_gives_nothing():
    assert await answer_side_question(Raw("Okay, the user is asking about Rome..."), "Is it nice?") is None


async def test_answer_cut_off_by_the_token_limit_is_saved():
    # from a real chat: the food answer ran on until the request timed out
    raw = '{"answer": "Try cacio e pepe and carbonara. Gelato is a must. You could also visit the mark'
    assert await answer_side_question(Raw(raw), "What food?") == (
        "Try cacio e pepe and carbonara. Gelato is a must."
    )


async def test_half_sentence_is_dropped():
    raw = '{"answer": "Try cacio e pepe. Also the famous Roman pizza al tag'
    assert await answer_side_question(Raw(raw), "What food?") == "Try cacio e pepe."


async def test_answer_length_is_capped():
    llm = Raw('{"answer": "Yes."}')
    await answer_side_question(llm, "Is it nice?")
    assert llm.max_tokens == 200
