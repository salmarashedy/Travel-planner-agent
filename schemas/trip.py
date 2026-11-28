from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.country import normalize_country

Interest = Literal["history", "food", "nature", "shopping", "adventure", "beaches"]
Currency = Literal["EUR", "USD", "EGP", "GBP"]
BudgetScope = Literal["total", "per_person"]
FieldName = Literal[
    "country",
    "region",
    "start_date",
    "num_days",
    "budget_amount",
    "budget_currency",
    "budget_scope",
    "party_size",
    "hotel_stars",
    "min_review_score",
]

ALLOWED_CURRENCIES: tuple[str, ...] = ("EUR", "USD", "EGP", "GBP")

# "$", "£", "dollars" and "pounds" are left out on purpose: they're ambiguous
CURRENCY_WORDS: dict[str, str] = {
    "€": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "us dollar": "USD",
    "us dollars": "USD",
    "egyptian pound": "EGP",
    "egyptian pounds": "EGP",
    "british pound": "GBP",
    "british pounds": "GBP",
    "pound sterling": "GBP",
    "sterling": "GBP",
}

# in the order the bot asks for them
REQUIRED_FIELDS: tuple[str, ...] = (
    "country",
    "start_date",
    "num_days",
    "budget_amount",
    "budget_currency",
    "party_size",
)


# past start dates are allowed here on purpose, so an old chat can still load;
# merge and decide check the date instead
class TripSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    country: str | None = None
    region: str | None = None
    start_date: date | None = None
    num_days: int | None = Field(default=None, ge=1, le=60)
    budget_amount: float | None = Field(default=None, gt=0)
    budget_currency: Currency | None = None
    budget_scope: BudgetScope = "total"
    party_size: int | None = Field(default=None, ge=1, le=20)
    hotel_stars: int | None = Field(default=None, ge=1, le=5)
    min_review_score: float | None = Field(default=None, ge=0, le=10)
    interests: list[Interest] = Field(default_factory=list)
    from_memory: list[str] = Field(default_factory=list)

    @field_validator("country")
    @classmethod
    def _normalize_country(cls, value: str | None) -> str | None:
        return None if value is None else normalize_country(value)

    @field_validator("region")
    @classmethod
    def _clean_region(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("budget_currency", mode="before")
    @classmethod
    def _normalize_currency(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        text = value.strip()
        return CURRENCY_WORDS.get(text.lower(), text.upper())

    def missing_required(self) -> list[str]:
        return [name for name in REQUIRED_FIELDS if getattr(self, name) is None]

    def is_complete(self) -> bool:
        return not self.missing_required()


# dates stay as text so one badly written date doesn't break the whole answer
class Updates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str | None = None
    region: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    num_days: int | None = None
    budget_amount: float | None = None
    budget_currency: str | None = None
    budget_scope: BudgetScope | None = None
    party_size: int | None = None
    hotel_stars: int | None = None
    min_review_score: float | None = None


class Conflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: FieldName
    options: list[str] = Field(min_length=2)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_type: Literal["trip_info", "greeting", "off_topic", "non_english", "yes", "no"]
    updates: Updates = Field(default_factory=Updates)
    interests_add: list[Interest] = Field(default_factory=list)
    interests_remove: list[Interest] = Field(default_factory=list)
    unclear: list[FieldName] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    side_question: str | None = None
