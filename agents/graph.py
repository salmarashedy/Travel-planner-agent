"""The Feature 1 chat turn as a LangGraph graph.

    extract --(model failed)--> fail --> END
       |
       v
     merge --> decide --> reply --> END

Each chat is saved by the checkpointer under its session id, so the next
message picks up where the last one stopped. Everything stored in the state
is plain JSON (dicts, lists, strings) so any checkpointer can save it.
"""

import operator
from collections.abc import Callable
from datetime import date
from typing import Annotated, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.decide import Decision, decide
from agents.extract import StructuredLLM, extract
from agents.grounding import tidy_extraction
from agents.merge import MergeResult, merge
from agents.reply import FALLBACK, build_reply
from schemas.trip import Extraction, TripSpec


class ChatState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # appended to, never replaced
    user_id: str
    spec: dict
    awaiting_confirmation: bool
    confirmed: bool
    ask_counts: dict[str, int]
    reply: str

    # Only meaningful during the current turn.
    prev_spec: dict
    extraction: dict | None
    rejected: dict[str, str]
    conflicts: list[str]
    side_question: str | None
    action: str
    fields: list[str]


def _spec(state: ChatState, key: str = "spec") -> TripSpec:
    return TripSpec(**state.get(key, {}))


def _dump(spec: TripSpec) -> dict:
    return spec.model_dump(mode="json")


def _user_message(state: ChatState) -> str:
    return state["messages"][-1]["content"]


def _last_question(state: ChatState) -> str | None:
    for message in reversed(state["messages"][:-1]):
        if message["role"] == "assistant":
            return message["content"]
    return None


def build_graph(
    llm: StructuredLLM,
    checkpointer: BaseCheckpointSaver | None = None,
    today: Callable[[], date] = date.today,
):
    async def extract_node(state: ChatState) -> dict:
        spec = _spec(state)
        found = await extract(
            llm,
            _user_message(state),
            spec,
            today(),
            last_question=_last_question(state),
            history=state["messages"][:-1],
        )
        if found is not None:
            found = tidy_extraction(found, _user_message(state))
        return {
            "spec": _dump(spec),
            "prev_spec": _dump(spec),
            "confirmed": state.get("confirmed", False),
            "extraction": found.model_dump(mode="json") if found else None,
            "rejected": {},
            "conflicts": [],
            "side_question": None,
            "fields": [],
        }

    def fail_node(state: ChatState) -> dict:
        # Saved in the history too, so the next turn knows what the bot said.
        return {
            "action": "fail",
            "reply": FALLBACK,
            "messages": [{"role": "assistant", "content": FALLBACK}],
        }

    def merge_node(state: ChatState) -> dict:
        result = merge(_spec(state), Extraction(**state["extraction"]), _user_message(state), today())
        return {
            "spec": _dump(result.spec),
            "rejected": result.rejected,
            "conflicts": result.conflicts,
            "side_question": result.side_question,
        }

    def decide_node(state: ChatState) -> dict:
        merged = MergeResult(spec=_spec(state), rejected=state["rejected"], conflicts=state["conflicts"])
        decision = decide(
            merged,
            _spec(state, "prev_spec"),
            Extraction(**state["extraction"]),
            state.get("awaiting_confirmation", False),
            today(),
            already_confirmed=state.get("confirmed", False),
        )
        return {
            "spec": _dump(decision.spec),
            "action": decision.action,
            "fields": decision.fields,
            "conflicts": decision.conflicts,
            "rejected": decision.rejected,
            "awaiting_confirmation": decision.awaiting_confirmation,
            "confirmed": decision.action == "confirmed",
        }

    def reply_node(state: ChatState) -> dict:
        extraction = Extraction(**state["extraction"])
        decision = Decision(
            action=state["action"],
            spec=_spec(state),
            fields=state["fields"],
            conflicts=state["conflicts"],
            rejected=state["rejected"],
            awaiting_confirmation=state["awaiting_confirmation"],
        )
        reply = build_reply(
            decision,
            _spec(state, "prev_spec"),
            extraction.message_type,
            state.get("ask_counts", {}),
            unclear=extraction.unclear,
            side_answer=None,  # step 4b: short side answers come later
            user_message=_user_message(state),
        )
        return {
            "reply": reply.text,
            "ask_counts": reply.ask_counts,
            "messages": [{"role": "assistant", "content": reply.text}],
        }

    graph = StateGraph(ChatState)
    graph.add_node("extract", extract_node)
    graph.add_node("fail", fail_node)
    graph.add_node("merge", merge_node)
    graph.add_node("decide", decide_node)
    graph.add_node("reply", reply_node)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges(
        "extract", lambda s: "fail" if s["extraction"] is None else "merge", ["fail", "merge"]
    )
    graph.add_edge("merge", "decide")
    graph.add_edge("decide", "reply")
    graph.add_edge("fail", END)
    graph.add_edge("reply", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


async def run_turn(graph, session_id: str, message: str, user_id: str | None = None) -> ChatState:
    """Send one user message through the graph and return the saved state."""
    turn: ChatState = {"messages": [{"role": "user", "content": message}]}
    if user_id:
        turn["user_id"] = user_id  # kept for long-term memory later
    return await graph.ainvoke(turn, config={"configurable": {"thread_id": session_id}})
