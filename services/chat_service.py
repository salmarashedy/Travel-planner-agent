import asyncio
import logging
from collections import defaultdict

from agents.graph import build_graph, run_turn
from core.config import Settings
from schemas.chat import ChatRequest, ChatResponse
from schemas.trip import TripSpec
from services.llm_client import OllamaClient

logger = logging.getLogger(__name__)

ERROR_REPLY = "Sorry, something went wrong on my side. Please try again."


class ChatService:
    def __init__(self, graph) -> None:
        self.graph = graph
        # one lock per session, otherwise two messages at once overwrite each other
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def handle(self, request: ChatRequest) -> ChatResponse:
        async with self._locks[request.session_id]:
            try:
                state = await run_turn(
                    self.graph, request.session_id, request.message, user_id=request.user_id
                )
            except Exception:
                logger.exception("chat turn failed for session %s", request.session_id)
                return ChatResponse(reply=ERROR_REPLY, trip_spec=None, status="error")

        return ChatResponse(
            reply=state["reply"],
            trip_spec=TripSpec(**state["spec"]),
            status="confirmed" if state.get("confirmed") else "collecting",
        )


def build_chat_service(settings: Settings) -> ChatService:
    return ChatService(build_graph(OllamaClient(settings)))
