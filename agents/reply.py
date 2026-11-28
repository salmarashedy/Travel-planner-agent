"""Turn a decision into the text the user sees. Fixed templates only, so the bot
can't invent facts or prices."""

from dataclasses import dataclass
from datetime import date

from agents.decide import Decision
from schemas.trip import TripSpec

FALLBACK = "Sorry, I didn't catch that. Could you rephrase?"
CONFIRMED = "Great, searching for hotels now..."
STRICT_AFTER = 3  # asks in a row before switching to a format hint

QUESTIONS = {
    "country": "Which country would you like to visit?",
    "region": "Which city or area would you like to stay in?",
    "start_date": "What day would you like the trip to start?",
    "num_days": "How many days will the trip be?",
    "budget_amount": "What is your budget for the trip?",
    "budget_currency": "Which currency is that in: EUR, USD, EGP or GBP?",
    "party_size": "How many people are travelling, counting children?",
    "hotel_stars": "How many stars would you like for the hotel (1 to 5)?",
    "min_review_score": "What is the lowest review score you'd accept (0 to 10)?",
}
VAGUE_QUESTIONS = {
    "country": "Which country exactly?",
    "start_date": "Which exact day? (for example, 10 June)",
    "num_days": "How many days exactly?",
    "budget_amount": "Roughly how much: under 1,000, 1,000 to 2,000, or more?",
    "budget_currency": "Which currency: EUR, USD, EGP or GBP?",
    "party_size": "Exactly how many people, counting children?",
}
STRICT_HINTS = {
    "country": "Please reply with just the country name, like Italy.",
    "start_date": "Please reply with a date, like 10 June 2027.",
    "budget_currency": "Please reply with one of: EUR, USD, EGP or GBP.",
}
NUMBER_HINT = "Please reply with just a number, like {example}."
NUMBER_EXAMPLES = {"num_days": "7", "budget_amount": "1500", "party_size": "2",
                   "hotel_stars": "4", "min_review_score": "8"}


@dataclass
class Reply:
    text: str
    ask_counts: dict[str, int]


def label(name: str) -> str:
    return name.replace("_", " ")


def show_number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def show_date(value: date) -> str:
    return f"{value.day} {value:%B %Y}"


def show_value(value) -> str:
    if isinstance(value, date):
        return show_date(value)
    if isinstance(value, (int, float)):
        return show_number(value)
    return str(value).replace("_", " ")


def question(name: str, times_asked: int, vague: bool) -> str:
    if times_asked > STRICT_AFTER:
        if name in NUMBER_EXAMPLES:
            return NUMBER_HINT.format(example=NUMBER_EXAMPLES[name])
        if name in STRICT_HINTS:
            return STRICT_HINTS[name]
    if vague and name in VAGUE_QUESTIONS:
        return VAGUE_QUESTIONS[name]
    return QUESTIONS.get(name, f"Could you tell me your preferred {label(name)}?")


def change_note(before: TripSpec, after: TripSpec) -> str | None:
    changes = []
    for name in QUESTIONS:
        old, new = getattr(before, name), getattr(after, name)
        if old is not None and new is not None and old != new:
            changes.append(f"{label(name)} {show_value(old)} -> {show_value(new)}")
    return f"Updated: {', '.join(changes)}." if changes else None


def summary(spec: TripSpec) -> str:
    assumed = " (assumed from your last trip)"

    def mark(text: str, name: str) -> str:
        return text + assumed if name in spec.from_memory else text

    parts = []
    if spec.country:
        parts.append(f"{spec.country} ({spec.region})" if spec.region else spec.country)
    if spec.num_days:
        days = f"{spec.num_days} day{'s' if spec.num_days != 1 else ''}"
        parts.append(f"{days} from {show_date(spec.start_date)}" if spec.start_date else days)
    if spec.budget_amount:
        scope = "per person" if spec.budget_scope == "per_person" else "in total"
        parts.append(f"{show_number(spec.budget_amount)} {spec.budget_currency or ''} {scope}".replace("  ", " "))
    if spec.party_size:
        parts.append(f"{spec.party_size} traveller{'s' if spec.party_size != 1 else ''}")
    if spec.hotel_stars:
        parts.append(mark(f"{spec.hotel_stars}-star hotels", "hotel_stars"))
    if spec.min_review_score is not None:
        parts.append(mark(f"reviews {spec.min_review_score:g}+", "min_review_score"))
    if spec.interests:
        parts.append(mark(f"interests: {', '.join(spec.interests)}", "interests"))
    return f"Here is what I have: {' | '.join(parts)}. Is this right?"


def build_reply(
    decision: Decision,
    before: TripSpec,
    message_type: str,
    ask_counts: dict[str, int],
    unclear: list[str] = (),
    side_answer: str | None = None,
) -> Reply:
    if decision.action == "confirmed":
        return Reply(CONFIRMED, {})

    asked = decision.fields if decision.action in ("ask", "redirect") else []
    # Counts are for asks in a row, so anything not asked this turn starts over.
    counts = {name: ask_counts.get(name, 0) + 1 for name in asked}
    questions = [question(name, counts[name], name in unclear) for name in asked]

    parts = []
    if message_type == "greeting":
        parts.append("Hello!")
    if side_answer:
        parts.append(side_answer)
    note = change_note(before, decision.spec)
    if note:
        parts.append(note)
    parts.extend(decision.rejected.values())

    if decision.action == "redirect":
        if message_type == "non_english":
            parts.append("For now I can only chat in English.")
        else:
            parts.append("I can only help with planning your trip.")
        parts.extend(questions)
    elif decision.action == "clarify":
        parts.extend(decision.conflicts)
        parts.append("Which one is right?")
    elif decision.action == "ask_change":
        parts.append("What would you like to change?")
    elif decision.action == "ask":
        parts.extend(questions)
    else:
        parts.append(summary(decision.spec))

    return Reply(" ".join(parts), counts)
