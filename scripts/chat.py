# Chat with the bot in the terminal (empty line to stop):
#   python -m scripts.chat
#   python -m scripts.chat "7 days in Italy" "two people"

import asyncio
import sys
import uuid
import warnings

# noisy deprecation warning from inside langgraph
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from agents.graph import build_graph, run_turn
from core.config import get_settings
from services.llm_client import OllamaClient


async def main(messages: list[str]) -> None:
    graph = build_graph(OllamaClient(get_settings()))
    session = f"terminal_{uuid.uuid4().hex[:8]}"

    async def send(text: str) -> None:
        state = await run_turn(graph, session, text)
        print(f"bot: {state['reply']}\n")
        if state.get("confirmed"):
            known = {k: v for k, v in state["spec"].items() if v not in (None, [], "total")}
            print(f"[confirmed] trip: {known}\n")

    if messages:
        for text in messages:
            print(f"you: {text}")
            await send(text)
        return

    print("Type a message (empty line to stop).\n")
    while text := input("you: ").strip():
        await send(text)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
