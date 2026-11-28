from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.trip import TripSpec


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=8, max_length=100)
    message: str = Field(min_length=1, max_length=1000)
    user_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    trip_spec: TripSpec | None
    status: Literal["collecting", "confirmed", "error"]
