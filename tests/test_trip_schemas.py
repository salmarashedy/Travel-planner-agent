from datetime import date

import pytest
from pydantic import ValidationError

from schemas.country import normalize_country
from schemas.trip import Extraction, TripSpec


# ---------- Country ----------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Italy", "Italy"),
        ("italy", "Italy"),
        ("  Spain ", "Spain"),
        ("IT", "Italy"),
        ("ITA", "Italy"),
        ("italia", "Italy"),
        ("France", "France"),
        ("Japan", "Japan"),
        ("Egypt", "Egypt"),
        ("USA", "United States"),
        ("UK", "United Kingdom"),
        ("England", "United Kingdom"),
        ("United Kingdom", "United Kingdom"),
        ("Britain", "United Kingdom"),
        ("Russia", "Russia"),
        ("Turkey", "Turkey"),
        ("Holland", "Netherlands"),
        ("UAE", "United Arab Emirates"),
        ("Ivory Coast", "Côte d'Ivoire"),
        ("Burma", "Myanmar"),
        ("Vatican", "Vatican City"),
        ("South Korea", "South Korea"),
    ],
)
def test_any_real_country_is_accepted_and_normalised(raw, expected):
    assert normalize_country(raw) == expected


@pytest.mark.parametrize("raw", ["Narnia", "Europe", "Paris", "Korea", "", "   "])
def test_unknown_or_ambiguous_country_is_refused(raw):
    with pytest.raises(ValueError):
        normalize_country(raw)


def test_trip_spec_normalises_country():
    assert TripSpec(country="usa").country == "United States"


def test_trip_spec_refuses_fake_country():
    with pytest.raises(ValidationError):
        TripSpec(country="Narnia")


# ---------- TripSpec rules ----------

def test_empty_form_is_missing_all_required_fields_in_order():
    assert TripSpec().missing_required() == [
        "country",
        "start_date",
        "num_days",
        "budget_amount",
        "budget_currency",
        "party_size",
    ]


def test_full_form_is_complete():
    spec = TripSpec(
        country="Italy",
        start_date=date(2027, 6, 10),
        num_days=7,
        budget_amount=1500,
        budget_currency="EUR",
        party_size=2,
    )
    assert spec.is_complete()
    assert spec.budget_scope == "total"


def test_currency_is_uppercased():
    assert TripSpec(budget_currency=" eur ").budget_currency == "EUR"


@pytest.mark.parametrize(
    ("word", "code"),
    [("euros", "EUR"), ("Euro", "EUR"), ("€", "EUR"), ("US dollars", "USD"),
     ("Egyptian pounds", "EGP"), ("sterling", "GBP")],
)
def test_clear_currency_words_become_codes(word, code):
    assert TripSpec(budget_currency=word).budget_currency == code


@pytest.mark.parametrize("word", ["$", "£", "dollars", "pounds"])
def test_unclear_currency_words_are_refused(word):
    with pytest.raises(ValidationError):
        TripSpec(budget_currency=word)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_days", 0),
        ("num_days", 61),
        ("budget_amount", 0),
        ("budget_amount", -100),
        ("budget_currency", "JPY"),
        ("party_size", 0),
        ("party_size", 21),
        ("hotel_stars", 6),
        ("min_review_score", 11),
        ("interests", ["skiing"]),
    ],
)
def test_out_of_range_values_are_refused(field, value):
    with pytest.raises(ValidationError):
        TripSpec(**{field: value})


def test_assignment_is_validated_so_bad_values_cannot_be_stored():
    spec = TripSpec(party_size=2)
    with pytest.raises(ValidationError):
        spec.party_size = -2
    assert spec.party_size == 2


def test_past_start_date_is_allowed_in_the_form_itself():
    # The date check lives in the merge step, so old sessions can still load.
    assert TripSpec(start_date=date(2000, 1, 1)).start_date == date(2000, 1, 1)


def test_unknown_field_is_refused():
    with pytest.raises(ValidationError):
        TripSpec(hotel_name="Hilton")


# ---------- Extraction ----------

def test_minimal_extraction_has_empty_defaults():
    ex = Extraction(message_type="yes")
    assert ex.updates.model_dump(exclude_none=True) == {}
    assert ex.unclear == [] and ex.conflicts == [] and ex.side_question is None


def test_extraction_parses_model_json():
    ex = Extraction.model_validate_json(
        '{"message_type": "trip_info",'
        ' "updates": {"country": "Italy", "num_days": 7, "start_date": "2027-06-10"},'
        ' "interests_add": ["history", "food"],'
        ' "unclear": ["budget_amount"],'
        ' "conflicts": [{"field": "num_days", "options": ["7", "10"]}]}'
    )
    assert ex.updates.country == "Italy"
    assert ex.updates.start_date == "2027-06-10"  # dates stay as text
    assert ex.interests_add == ["history", "food"]
    assert ex.conflicts[0].field == "num_days"


@pytest.mark.parametrize(
    "bad_json",
    [
        '{"message_type": "maybe"}',
        '{"message_type": "trip_info", "updates": {"hotel_name": "Hilton"}}',
        '{"message_type": "trip_info", "unclear": ["favourite_colour"]}',
        '{"message_type": "trip_info", "interests_add": ["skiing"]}',
        '{"message_type": "trip_info", "conflicts": [{"field": "num_days", "options": ["7"]}]}',
    ],
)
def test_extraction_refuses_invented_values(bad_json):
    with pytest.raises(ValidationError):
        Extraction.model_validate_json(bad_json)


def test_extraction_schema_is_json_serialisable_for_ollama():
    import json

    schema = Extraction.model_json_schema()
    assert json.loads(json.dumps(schema))["properties"]["message_type"]
