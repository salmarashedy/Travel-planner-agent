import pytest

from agents.grounding import Message, correct_message_type, unclear_mentioned


def test_unclear_needs_the_topic_in_the_message():
    msg = Message("i want to travel for 5 days in decamber")
    assert unclear_mentioned("start_date", msg)
    assert not unclear_mentioned("budget_amount", msg)
    assert not unclear_mentioned("budget_currency", msg)


def test_unclear_budget_when_the_user_is_vague_about_money():
    assert unclear_mentioned("budget_amount", Message("something cheap in June"))


@pytest.mark.parametrize("message", ["okay", "OK", "yes", "yep", "sure", "looks good", "That's right!"])
def test_short_yes_replies(message):
    assert correct_message_type("greeting", message) == "yes"


@pytest.mark.parametrize("message", ["no", "Nope", "nah", "wrong"])
def test_short_no_replies(message):
    assert correct_message_type("off_topic", message) == "no"


@pytest.mark.parametrize("message", ["nth", "nothing", "nvm", "never mind"])
def test_nothing_to_change_carries_on(message):
    assert correct_message_type("non_english", message) == "trip_info"


def test_no_at_the_start_is_a_no():
    assert correct_message_type("trip_info", "no i said me and my husband and 3 childern") == "no"


def test_no_at_the_start_does_not_hide_off_topic():
    assert correct_message_type("off_topic", "no idea what the capital of France is") == "off_topic"


def test_ascii_text_is_not_non_english():
    assert correct_message_type("non_english", "idk lol") == "off_topic"


def test_real_non_english_stays():
    assert correct_message_type("non_english", "Quiero ir a España") == "non_english"


def test_non_english_without_accents_stays():
    # caught by the held-out set
    assert correct_message_type("non_english", "Bonjour, je veux aller en Espagne") == "non_english"


def test_longer_messages_keep_the_models_label():
    assert correct_message_type("trip_info", "okay so 5 days in Rome") == "trip_info"
