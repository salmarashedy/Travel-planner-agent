import json
from datetime import date

import pytest

from agents.graph import build_graph, run_turn
from agents.reply import CONFIRMED, FALLBACK, QUESTIONS
from core.exceptions import LLMClientError
from tests.fakes import CHAT as ANSWERS
from tests.fakes import FakeLLM, trip

TODAY = date(2026, 11, 21)


@pytest.fixture
def chat():
    llm = FakeLLM(dict(ANSWERS))
    graph = build_graph(llm, today=lambda: TODAY)

    async def say(message, session="session_aaa"):
        return await run_turn(graph, session, message)

    say.llm = llm
    return say


async def test_the_four_turn_example_from_the_guide(chat):
    s = await chat("7 days in Italy, 1500 euros, 4 stars")
    assert s["reply"] == f"{QUESTIONS['start_date']} {QUESTIONS['party_size']}"
    assert s["confirmed"] is False

    s = await chat("June 10, 2 people")
    assert s["reply"].startswith("Here is what I have: Italy | 7 days from 10 June 2027")
    assert s["awaiting_confirmation"] is True

    s = await chat("actually make it 2000")
    assert s["reply"].startswith("Updated: budget amount 1,500 -> 2,000. Here is what I have:")

    s = await chat("yes")
    assert s["reply"] == CONFIRMED
    assert s["confirmed"] is True
    assert s["spec"]["budget_amount"] == 2000
    assert s["spec"]["country"] == "Italy"
    assert len(s["messages"]) == 8


async def test_sessions_do_not_mix(chat):
    await chat("7 days in Italy, 1500 euros, 4 stars", session="session_aaa")
    other = await chat("hi", session="session_bbb")
    assert other["spec"]["country"] is None
    assert len(other["messages"]) == 2


async def test_model_down_gives_a_friendly_reply_and_the_chat_goes_on(chat):
    chat.llm.answers["7 days in Italy, 1500 euros, 4 stars"] = LLMClientError("down")
    s = await chat("7 days in Italy, 1500 euros, 4 stars")
    assert s["reply"] == FALLBACK
    assert s["messages"][-1] == {"role": "assistant", "content": FALLBACK}
    assert s["confirmed"] is False
    assert s["spec"]["country"] is None

    chat.llm.answers["7 days in Italy, 1500 euros, 4 stars"] = ANSWERS["7 days in Italy, 1500 euros, 4 stars"]
    s = await chat("7 days in Italy, 1500 euros, 4 stars")
    assert s["spec"]["country"] == "Italy"


async def test_the_bots_last_question_is_given_to_the_model(chat):
    await chat("7 days in Italy, 1500 euros, 4 stars")
    await chat("two")
    system_prompt = chat.llm.calls[-1][0]["content"]
    assert QUESTIONS["party_size"] in system_prompt


async def test_off_topic_after_the_summary_then_yes_confirms(chat):
    await chat("7 days in Italy, 1500 euros, 4 stars")
    await chat("June 10, 2 people")

    s = await chat("what's the capital of France?")
    assert s["reply"] == "I can only help with planning your trip."
    assert s["awaiting_confirmation"] is True

    s = await chat("yes")
    assert s["confirmed"] is True


async def test_no_after_the_summary(chat):
    await chat("7 days in Italy, 1500 euros, 4 stars")
    await chat("June 10, 2 people")
    s = await chat("no")
    assert s["reply"] == "What would you like to change?"
    assert s["awaiting_confirmation"] is False


async def test_yes_before_anything_is_known_does_not_confirm(chat):
    s = await chat("yes")
    assert s["confirmed"] is False
    assert s["reply"] == f"{QUESTIONS['country']} {QUESTIONS['start_date']}"


async def test_old_chat_with_a_passed_start_date():
    llm = FakeLLM(dict(ANSWERS))
    clock = {"today": TODAY}
    graph = build_graph(llm, today=lambda: clock["today"])

    await run_turn(graph, "session_old", "7 days in Italy, 1500 euros, 4 stars")
    await run_turn(graph, "session_old", "June 10, 2 people")

    clock["today"] = date(2027, 6, 20)
    s = await run_turn(graph, "session_old", "yes")
    assert s["confirmed"] is False
    assert s["spec"]["start_date"] is None
    assert "That start date has now passed." in s["reply"]


async def test_your_chat_after_the_summary(chat):
    # the end of a real chat, with the labels qwen3:4b gave
    chat.llm.answers.update({
        "no i said me and my husband and 3 childern": trip(party_size=4),
        "nth": {"message_type": "non_english"},
        "okay": {"message_type": "greeting"},
        "June 10, my husband and 3 sons": trip(start_date="2027-06-10", party_size=4),
    })
    await chat("7 days in Italy, 1500 euros, 4 stars")
    await chat("June 10, my husband and 3 sons")

    s = await chat("no i said me and my husband and 3 childern")
    assert s["reply"] == "What would you like to change?"

    s = await chat("nth")
    assert s["reply"].startswith("Here is what I have:")

    s = await chat("okay")
    assert s["confirmed"] is True

    s = await chat("yes")
    assert s["reply"] == CONFIRMED
    assert s["confirmed"] is True


async def test_side_question_is_answered_and_the_chat_goes_on(chat):
    await chat("7 days in Italy, 1500 euros, 4 stars")
    s = await chat("is Rome nice in winter?")
    assert s["reply"] == (
        f"It's lovely that time of year. {QUESTIONS['start_date']} {QUESTIONS['party_size']}"
    )
    assert s["spec"]["region"] == "Rome"
    assert chat.llm.side_calls[-1][-1]["content"] == "Is Rome nice in winter?"


async def test_side_answer_with_a_price_is_replaced(chat):
    chat.llm.side_answers["Is Rome nice in winter?"] = "Yes, and hotels are about 80 euros."
    s = await chat("is Rome nice in winter?")
    assert s["reply"].startswith("I'll check live prices and details when I build your plan.")


async def test_side_answer_failing_still_gives_the_normal_reply(chat):
    chat.llm.side_answers["Is Rome nice in winter?"] = LLMClientError("down")
    s = await chat("is Rome nice in winter?")
    assert s["reply"] == f"{QUESTIONS['start_date']} {QUESTIONS['num_days']}"


async def test_side_questions_do_not_count_as_ignoring_ours(chat):
    # from a real chat: three side questions made the bot switch to strict hints
    await chat("7 days in Italy, 1500 euros, 4 stars")
    for _ in range(4):
        s = await chat("is Rome nice in winter?")
    assert QUESTIONS["start_date"] in s["reply"]
    assert "Please reply with" not in s["reply"]


async def test_no_side_call_without_a_question(chat):
    await chat("7 days in Italy, 1500 euros, 4 stars")
    assert chat.llm.side_calls == []


async def test_state_is_plain_json(chat):
    s = await chat("7 days in Italy, 1500 euros, 4 stars")
    json.dumps(s)
