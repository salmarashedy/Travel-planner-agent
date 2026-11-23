from typing import Any

from pydantic import ConfigDict, Field


class FlexibleModel:
    model_config = ConfigDict(extra="allow")


TripSpec = dict[str, Any]


class ErrorResponse(FlexibleModel):
    detail: str = Field(min_length=1)
