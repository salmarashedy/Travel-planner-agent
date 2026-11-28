import asyncio
import json


class FakeLLM:
    """Stands in for Qwen3: answers by message text, so tests read like a real chat.

    An answer can be a dict (returned as JSON) or an exception (raised).
    """

    def __init__(self, answers: dict, delay: float = 0):
        self.answers = answers
        self.delay = delay
        self.calls = []

    async def chat_structured(self, messages, schema):
        self.calls.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        answer = self.answers[messages[-1]["content"]]
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer)


def trip(**updates) -> dict:
    return {"message_type": "trip_info", "updates": updates}


CHAT = {
    "7 days in Italy, 1500 euros, 4 stars": trip(
        country="Italy", num_days=7, budget_amount=1500, budget_currency="EUR", hotel_stars=4),
    "June 10, 2 people": trip(start_date="2027-06-10", party_size=2),
    "actually make it 2000": trip(budget_amount=2000),
    "yes": {"message_type": "yes"},
    "no": {"message_type": "no"},
    "what's the capital of France?": {"message_type": "off_topic"},
    "two": trip(party_size=2),
    "hi": {"message_type": "greeting"},
}
