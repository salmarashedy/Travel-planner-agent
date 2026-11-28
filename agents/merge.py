"""Decide which of the model's findings are allowed into the trip form.

The model only suggests values. This step checks each one and keeps the
good ones, so one bad value never wipes out the rest of the message.
"""

import logging
from dataclasses import dataclass, field
from datetime import date

from pydantic import ValidationError

from agents.grounding import (
    Message,
    date_mentioned,
    either_or,
    interest_mentioned,
    is_grounded,
    question_mentioned,
)
from schemas.trip import Extraction, TripSpec

logger = logging.getLogger(__name__)

MAX_DAYS_AHEAD = 365

REASONS = {
    "num_days": "Trips can be from 1 to 60 days.",
    "budget_amount": "The budget needs to be more than 0.",
    "budget_currency": "I can work with EUR, USD, EGP or GBP.",
    "party_size": "The group can be from 1 to 20 people.",
    "hotel_stars": "Hotel stars go from 1 to 5.",
    "min_review_score": "Review scores go from 0 to 10.",
}


@dataclass
class MergeResult:
    spec: TripSpec
    rejected: dict[str, str] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    side_question: str | None = None
    dropped: list[str] = field(default_factory=list)  # values the user never said


def _label(name: str) -> str:
    return name.replace("_", " ")


def _reason(name: str, error: ValidationError) -> str:
    if name in REASONS:
        return REASONS[name]
    # Our own validators (like the country check) already have a friendly message.
    return error.errors()[0]["msg"].removeprefix("Value error, ")


def merge(spec: TripSpec, extraction: Extraction, message: str, today: date) -> MergeResult:
    result = MergeResult(spec=spec)
    if extraction.message_type in ("off_topic", "non_english"):
        return result

    msg = Message(message)
    updates = extraction.updates.model_dump(exclude_none=True)

    for conflict in extraction.conflicts:
        said = [o for o in conflict.options if o.lower() in msg.text]
        if len(said) < 2:
            continue  # the model imagined it
        updates.pop(conflict.field, None)
        result.conflicts.append(f"For {_label(conflict.field)} you said {' and '.join(said)}.")

    # The model often just picks one side of "1500 or 2000"; treat it as the conflict it is.
    for name in ("budget_amount", "num_days", "party_size", "hotel_stars"):
        options = either_or(updates[name], msg) if name in updates else None
        if options:
            updates.pop(name)
            result.conflicts.append(f"For {_label(name)} you said {' and '.join(options)}.")

    region_kept = "region" in updates and is_grounded("region", updates["region"], msg)
    for name in list(updates):
        if name in ("start_date", "end_date"):
            continue  # checked in _check_dates once they are parsed
        if not is_grounded(name, updates[name], msg, region_kept=region_kept):
            result.dropped.append(f"{name}={updates.pop(name)!r}")

    # After grounding: num_days worked out from the dates is ours, not the model's.
    _check_dates(updates, msg, today, spec, result)

    new = spec.model_copy(deep=True)
    for name, value in updates.items():
        try:
            setattr(new, name, value)
        except ValidationError as error:
            result.rejected[name] = _reason(name, error)
            continue
        if name in new.from_memory:
            new.from_memory.remove(name)

    added = [i for i in extraction.interests_add if interest_mentioned(i, msg)]
    removed = [i for i in extraction.interests_remove if interest_mentioned(i, msg)]
    for i in extraction.interests_add + extraction.interests_remove:
        if i not in added and i not in removed:
            result.dropped.append(f"interest={i!r}")
    if added or removed:
        interests = [i for i in new.interests if i not in removed]
        interests += [i for i in added if i not in interests and i not in removed]
        new.interests = interests
        if "interests" in new.from_memory:
            new.from_memory.remove("interests")

    question = extraction.side_question
    if question and extraction.message_type == "trip_info" and question_mentioned(question, msg):
        result.side_question = question
    elif question:
        result.dropped.append(f"side_question={question!r}")

    if result.dropped:
        logger.info("dropped values the user did not say: %s", ", ".join(result.dropped))
    result.spec = new
    return result


def _check_dates(updates: dict, msg: Message, today: date, spec: TripSpec, result: MergeResult) -> None:
    parsed = {}
    for key in ("start_date", "end_date"):
        if key not in updates:
            continue
        text = updates.pop(key)
        try:
            value = date.fromisoformat(text)
        except ValueError:
            # An unreadable end date is reported under num_days so the bot asks
            # "what is your end date, or how many days?" and not for the start again.
            result.rejected["num_days" if key == "end_date" else "start_date"] = "I couldn't read that date."
            continue
        if not date_mentioned(value, msg):
            result.dropped.append(f"{key}={text!r}")
            continue
        parsed[key] = value

    start = parsed.get("start_date")
    if start is not None:
        if start < today:
            result.rejected["start_date"] = "That start date is in the past."
            start = None
        elif (start - today).days > MAX_DAYS_AHEAD:
            result.rejected["start_date"] = "That date is more than a year away."
            start = None
        else:
            updates["start_date"] = start

    end = parsed.get("end_date")
    if end is None:
        return
    start = start or spec.start_date
    if start is None:
        if "start_date" not in result.rejected:
            result.rejected["start_date"] = "I need the start date to work out the trip length."
        return
    if end < start:
        result.rejected["num_days"] = "The end date is before the start date."
        return

    days = (end - start).days + 1
    said = updates.get("num_days")
    if said is not None and said != days:
        updates.pop("num_days")
        result.conflicts.append(f"The dates you gave cover {days} days, but you said {said}.")
    else:
        updates["num_days"] = days
