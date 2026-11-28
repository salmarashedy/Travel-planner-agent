import asyncio
import json


# answers by message text; an answer can also be an exception to raise
class FakeLLM:
    def __init__(self, answers: dict, delay: float = 0, side_answers: dict | None = None):
        self.answers = answers
        self.side_answers = side_answers or {}
        self.delay = delay
        self.calls = []
        self.side_calls = []

    async def chat_structured(self, messages, schema, max_tokens=None):
        if schema.get("title") == "SideAnswer":
            self.side_calls.append(messages)
            answer = self.side_answers.get(messages[-1]["content"], "It's lovely that time of year.")
            if isinstance(answer, Exception):
                raise answer
            return json.dumps({"answer": answer})

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
    "is Rome nice in winter?": {
        "message_type": "trip_info",
        "updates": {"region": "Rome", "country": "Italy"},
        "side_question": "Is Rome nice in winter?",
    },
}
