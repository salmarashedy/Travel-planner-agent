from datetime import date

import pytest

from agents.decide import Decision
from agents.reply import CONFIRMED, QUESTIONS, build_reply, change_note, question, summary
from schemas.trip import FieldName, TripSpec

FULL = TripSpec(
    country="Italy",
    start_date=date(2027, 6, 10),
    num_days=7,
    budget_amount=1500,
    budget_currency="EUR",
    party_size=2,
)


def reply(action, spec=None, *, before=None, message_type="trip_info", counts=None, **kwargs):
    unclear = kwargs.pop("unclear", [])
    side_answer = kwargs.pop("side_answer", None)
    decision = Decision(action=action, spec=spec or TripSpec(), **kwargs)
    return build_reply(decision, before or decision.spec, message_type, counts or {},
                       unclear=unclear, side_answer=side_answer)


# ---------- questions ----------

def test_every_field_has_a_question():
    from typing import get_args
    for name in get_args(FieldName):
        assert question(name, 1, vague=False)


def test_field_without_a_question_gets_a_general_one():
    assert question("budget_scope", 1, vague=False) == "Could you tell me your preferred budget scope?"


def test_empty_form_asks_two_questions():
    r = reply("ask", fields=["country", "start_date"])
    assert r.text == f"{QUESTIONS['country']} {QUESTIONS['start_date']}"
    assert r.ask_counts == {"country": 1, "start_date": 1}


def test_vague_answer_gets_a_more_helpful_question():
    r = reply("ask", fields=["budget_amount"], unclear=["budget_amount"])
    assert r.text == "Roughly how much: under 1,000, 1,000 to 2,000, or more?"


def test_asked_too_many_times_gets_a_format_hint():
    r = reply("ask", fields=["budget_amount"], counts={"budget_amount": 3})
    assert r.text == "Please reply with just a number, like 1500."
    assert r.ask_counts == {"budget_amount": 4}


def test_third_time_is_still_the_normal_question():
    r = reply("ask", fields=["party_size"], counts={"party_size": 2})
    assert r.text == QUESTIONS["party_size"]


def test_counts_reset_for_fields_not_asked():
    r = reply("ask", fields=["party_size"], counts={"party_size": 2, "country": 5})
    assert r.ask_counts == {"party_size": 3}


# ---------- summary ----------

def test_summary_of_a_full_form():
    assert summary(FULL) == (
        "Here is what I have: Italy | 7 days from 10 June 2027 | 1,500 EUR in total | "
        "2 travellers. Is this right?"
    )


def test_summary_with_everything():
    spec = FULL.model_copy(update={
        "region": "Rome", "budget_scope": "per_person", "party_size": 1, "num_days": 1,
        "hotel_stars": 4, "min_review_score": 8.5, "interests": ["history", "food"],
    })
    assert summary(spec) == (
        "Here is what I have: Italy (Rome) | 1 day from 10 June 2027 | 1,500 EUR per person | "
        "1 traveller | 4-star hotels | reviews 8.5+ | interests: history, food. Is this right?"
    )


def test_summary_never_says_none():
    assert "None" not in summary(TripSpec(country="Spain"))
    assert "None" not in summary(FULL)


def test_summary_marks_memory_values():
    spec = FULL.model_copy(update={"hotel_stars": 4, "from_memory": ["hotel_stars"]})
    assert "4-star hotels (assumed from your last trip)" in summary(spec)


def test_confirm_shows_the_summary():
    assert reply("confirm", FULL).text == summary(FULL)


# ---------- change note ----------

def test_change_note_formats_numbers():
    after = FULL.model_copy(update={"budget_amount": 2000})
    assert change_note(FULL, after) == "Updated: budget amount 1,500 -> 2,000."


def test_change_note_formats_dates():
    after = FULL.model_copy(update={"start_date": date(2027, 7, 1)})
    assert change_note(FULL, after) == "Updated: start date 10 June 2027 -> 1 July 2027."


def test_no_change_note_when_a_field_is_filled_for_the_first_time():
    assert change_note(TripSpec(), FULL) is None


def test_change_then_summary_like_the_guide_example():
    after = FULL.model_copy(update={"budget_amount": 2000})
    r = reply("confirm", after, before=FULL)
    assert r.text.startswith("Updated: budget amount 1,500 -> 2,000. Here is what I have:")
    assert "2,000 EUR in total" in r.text


# ---------- other actions ----------

def test_confirmed():
    r = reply("confirmed", FULL)
    assert r.text == CONFIRMED
    assert r.ask_counts == {}


def test_clarify():
    r = reply("clarify", conflicts=["For budget amount you said 1500 and 2000."])
    assert r.text == "For budget amount you said 1500 and 2000. Which one is right?"


def test_ask_change():
    assert reply("ask_change", FULL).text == "What would you like to change?"


def test_redirect_adds_the_pending_question():
    r = reply("redirect", fields=["party_size"], message_type="off_topic")
    assert r.text == f"I can only help with planning your trip. {QUESTIONS['party_size']}"


def test_redirect_with_nothing_pending():
    assert reply("redirect", FULL, message_type="off_topic").text == "I can only help with planning your trip."


def test_non_english_redirect():
    r = reply("redirect", fields=["country"], message_type="non_english")
    assert r.text.startswith("For now I can only chat in English.")


# ---------- putting the parts together ----------

def test_parts_come_in_order():
    before = TripSpec(country="Italy", budget_amount=1500)
    after = before.model_copy(update={"budget_amount": 2000})
    r = reply(
        "ask", after, before=before, message_type="greeting",
        fields=["hotel_stars"], rejected={"hotel_stars": "Hotel stars go from 1 to 5."},
        side_answer="Rome is busy in summer.",
    )
    assert r.text == (
        "Hello! Rome is busy in summer. Updated: budget amount 1,500 -> 2,000. "
        f"Hotel stars go from 1 to 5. {QUESTIONS['hotel_stars']}"
    )


@pytest.mark.parametrize("action", ["ask", "confirm", "clarify", "ask_change", "redirect"])
def test_no_reply_is_empty(action):
    assert reply(action, FULL, fields=["country"], conflicts=["x"]).text.strip()
