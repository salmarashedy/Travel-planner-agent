from datetime import date

from agents.decide import decide
from agents.merge import MergeResult, merge
from schemas.trip import Extraction, TripSpec

TODAY = date(2026, 11, 21)

FULL = TripSpec(
    country="Italy",
    start_date=date(2027, 6, 10),
    num_days=7,
    budget_amount=1500,
    budget_currency="EUR",
    party_size=2,
)


def said(message_type="trip_info", **kwargs) -> Extraction:
    return Extraction(message_type=message_type, **kwargs)


def run(spec, extraction=None, *, prev=None, awaiting=False, rejected=None, conflicts=None):
    merged = MergeResult(spec=spec, rejected=rejected or {}, conflicts=conflicts or [])
    return decide(merged, prev if prev is not None else spec, extraction or said(), awaiting, TODAY)


# ---------- asking ----------

def test_empty_form_asks_for_country_and_start_date():
    d = run(TripSpec())
    assert d.action == "ask"
    assert d.fields == ["country", "start_date"]
    assert d.awaiting_confirmation is False


def test_missing_fields_are_asked_in_order():
    spec = FULL.model_copy(update={"num_days": None, "party_size": None, "budget_currency": None})
    assert run(spec).fields == ["num_days", "budget_currency"]


def test_refused_value_is_asked_first():
    d = run(FULL, rejected={"hotel_stars": "Hotel stars go from 1 to 5."})
    assert (d.action, d.fields) == ("ask", ["hotel_stars"])
    assert d.rejected == {"hotel_stars": "Hotel stars go from 1 to 5."}


def test_vague_value_comes_before_other_missing_fields():
    spec = TripSpec(country="Italy")
    d = run(spec, said(unclear=["budget_amount"]))
    assert d.fields == ["budget_amount", "start_date"]


def test_unclear_flag_on_a_known_field_is_ignored():
    d = run(FULL, said(unclear=["budget_currency"]))
    assert d.action == "confirm"


# ---------- summary and confirmation ----------

def test_full_form_shows_the_summary():
    d = run(FULL, prev=TripSpec())
    assert d.action == "confirm"
    assert d.awaiting_confirmation is True


def test_yes_to_the_summary_confirms():
    d = run(FULL, said("yes"), awaiting=True)
    assert d.action == "confirmed"
    assert d.awaiting_confirmation is False


def test_yes_without_a_summary_does_not_confirm():
    d = run(FULL, said("yes"), awaiting=False)
    assert d.action == "confirm"
    assert d.awaiting_confirmation is True


def test_yes_but_change_shows_the_summary_again():
    new = FULL.model_copy(update={"hotel_stars": 5})
    d = run(new, said("trip_info"), prev=FULL, awaiting=True)
    assert d.action == "confirm"
    assert d.awaiting_confirmation is True


def test_yes_with_a_real_change_does_not_confirm():
    # Even if the model labels it "yes", a changed form needs a new summary.
    new = FULL.model_copy(update={"budget_amount": 2000})
    assert run(new, said("yes"), prev=FULL, awaiting=True).action == "confirm"


def test_model_repeating_a_known_value_still_confirms():
    # The model said country=Italy again on "yes", but the form is the same.
    same = FULL.model_copy(update={"country": "Italy"})
    assert run(same, said("yes"), prev=FULL, awaiting=True).action == "confirmed"


def test_yes_with_a_conflict_does_not_confirm():
    d = run(FULL, said("yes"), awaiting=True, conflicts=["For budget amount you said 1500 and 2000."])
    assert d.action == "clarify"


def test_no_to_the_summary_asks_what_to_change():
    d = run(FULL, said("no"), awaiting=True)
    assert d.action == "ask_change"
    assert d.awaiting_confirmation is False


def test_no_without_a_summary_just_continues():
    assert run(TripSpec(country="Italy"), said("no")).action == "ask"


# ---------- off topic ----------

def test_off_topic_keeps_its_place():
    d = run(FULL.model_copy(update={"party_size": None}), said("off_topic"))
    assert d.action == "redirect"
    assert d.fields == ["party_size"]


def test_off_topic_after_the_summary_then_yes_confirms_in_one_go():
    first = run(FULL, said("off_topic"), awaiting=True)
    assert first.action == "redirect"
    assert first.fields == []
    assert first.awaiting_confirmation is True

    second = run(FULL, said("yes"), awaiting=first.awaiting_confirmation)
    assert second.action == "confirmed"


def test_non_english_is_redirected():
    assert run(TripSpec(), said("non_english")).action == "redirect"


# ---------- conflicts ----------

def test_conflict_is_clarified_before_asking_anything_else():
    d = run(TripSpec(), conflicts=["For budget amount you said 1500 and 2000."])
    assert d.action == "clarify"
    assert d.conflicts == ["For budget amount you said 1500 and 2000."]
    assert d.awaiting_confirmation is False


# ---------- old sessions ----------

def test_start_date_that_has_passed_is_cleared_and_asked_again():
    old = FULL.model_copy(update={"start_date": date(2026, 11, 1)})
    d = run(old, said("yes"), awaiting=True)
    assert d.action == "ask"
    assert d.fields == ["start_date"]
    assert d.spec.start_date is None
    assert d.rejected["start_date"] == "That start date has now passed."
    assert old.start_date == date(2026, 11, 1)  # the input is not changed


# ---------- the 4-turn example from the Feature 1 guide ----------

def test_four_turn_conversation():
    spec, awaiting = TripSpec(), False

    def turn(message, extraction):
        nonlocal spec, awaiting
        merged = merge(spec, extraction, message, TODAY)
        d = decide(merged, spec, extraction, awaiting, TODAY)
        spec, awaiting = d.spec, d.awaiting_confirmation
        return d

    d = turn("7 days in Italy, 1500 euros, 4 stars",
             said(updates={"country": "Italy", "num_days": 7, "budget_amount": 1500,
                           "budget_currency": "EUR", "hotel_stars": 4}))
    assert (d.action, d.fields) == ("ask", ["start_date", "party_size"])

    d = turn("June 10, 2 people", said(updates={"start_date": "2027-06-10", "party_size": 2}))
    assert d.action == "confirm" and awaiting

    d = turn("actually make it 2000", said(updates={"budget_amount": 2000}))
    assert d.action == "confirm" and spec.budget_amount == 2000 and awaiting

    d = turn("yes", said("yes"))
    assert d.action == "confirmed"
    assert spec.country == "Italy" and spec.budget_amount == 2000
