from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from agents.merge import MergeResult
from schemas.trip import REQUIRED_FIELDS, Extraction, TripSpec

Action = Literal["confirmed", "redirect", "clarify", "ask_change", "ask", "confirm"]

MAX_QUESTIONS = 2


@dataclass
class Decision:
    action: Action
    spec: TripSpec
    fields: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    awaiting_confirmation: bool = False


def open_needs(spec: TripSpec, rejected: dict[str, str], unclear: list[str]) -> list[str]:
    # only required fields count as vague; the model flags optional ones a lot
    vague = [n for n in unclear if n in REQUIRED_FIELDS and getattr(spec, n) is None]
    return list(dict.fromkeys([*rejected, *vague, *spec.missing_required()]))


def decide(
    merged: MergeResult,
    prev_spec: TripSpec,
    extraction: Extraction,
    awaiting: bool,
    today: date,
    already_confirmed: bool = False,
) -> Decision:
    spec = merged.spec
    rejected = dict(merged.rejected)
    changed = spec != prev_spec

    # old chat reopened after the start date
    if spec.start_date is not None and spec.start_date < today:
        spec = spec.model_copy(update={"start_date": None})
        rejected["start_date"] = "That start date has now passed."

    needs = open_needs(spec, rejected, extraction.unclear)
    kind = extraction.message_type

    def result(action: Action, **kwargs) -> Decision:
        return Decision(action=action, spec=spec, rejected=rejected, **kwargs)

    settled = not needs and not merged.conflicts and not changed
    if awaiting and kind == "yes" and settled:
        return result("confirmed")
    if already_confirmed and settled and kind in ("yes", "trip_info", "greeting"):
        return result("confirmed")

    if kind in ("off_topic", "non_english"):
        # keep the flag so a "yes" after a side remark still confirms
        return result("redirect", fields=needs[:1], awaiting_confirmation=awaiting)
    if merged.conflicts:
        return result("clarify", conflicts=list(merged.conflicts))
    if (awaiting or already_confirmed) and kind == "no" and not changed:
        return result("ask_change")
    if needs:
        return result("ask", fields=needs[:MAX_QUESTIONS])
    return result("confirm", awaiting_confirmation=True)
