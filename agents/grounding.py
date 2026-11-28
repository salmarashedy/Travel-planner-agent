import re
from datetime import date

from schemas.country import COUNTRY_ALIASES, normalize_country
from schemas.trip import CURRENCY_WORDS

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "solo": 1, "alone": 1, "couple": 2, "pair": 2, "both": 2,
}
PEOPLE_WORDS = {
    "me", "myself", "wife", "husband", "partner", "girlfriend", "boyfriend", "kid",
    "kids", "child", "children", "son", "sons", "daughter", "daughters", "baby",
    "friend", "friends", "family", "adult", "adults", "people", "person", "persons",
    "traveller", "travellers", "traveler", "travelers", "parents", "mom", "dad",
    "mother", "father", "brother", "sister", "group",
}
PER_PERSON_PHRASES = ("each", "per person", "pp", "a head", "per head", "apiece")
TOTAL_PHRASES = ("total", "altogether", "overall", "for all of us", "for everyone")
REVIEW_WORDS = ("review", "rating", "rated", "score")
RELATIVE_DATE_WORDS = {"today", "tomorrow", "next", "weekend", "week", "christmas", "easter"}
INTEREST_WORDS = {
    "history": {"history", "historic", "historical", "museum", "museums", "ruins",
                "castle", "castles", "ancient", "heritage", "culture", "cultural"},
    "food": {"food", "foodie", "eat", "eating", "cuisine", "restaurant", "restaurants",
             "dining", "wine"},
    "nature": {"nature", "hike", "hiking", "mountain", "mountains", "park", "parks",
               "forest", "lake", "lakes", "outdoors", "wildlife"},
    "shopping": {"shopping", "shop", "shops", "mall", "malls", "market", "markets", "boutiques"},
    "adventure": {"adventure", "adventures", "adventurous", "diving", "rafting",
                  "climbing", "skydiving", "safari"},
    "beaches": {"beach", "beaches", "seaside", "coast", "island", "islands", "swimming"},
}
NOT_MONEY = (
    "stars?|days?|nights?|weeks?|people|persons?|adults?|kids?|children|"
    "travell?ers?|of us|am|pm|th|st|nd|rd"
)
QUESTION_WORDS = {"is", "are", "what", "how", "can", "do", "does", "should", "when",
                  "where", "which", "why", "will", "would", "could"}
STOP_WORDS = {"the", "a", "an", "is", "are", "it", "in", "on", "to", "for", "of", "and",
              "or", "do", "does", "can", "what", "how", "there", "this", "that", "with"}


class Message:
    def __init__(self, text: str) -> None:
        self.raw = text
        # "1,500" -> "1500", but leave "Rome, Italy" alone
        self.text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
        self.words = set(re.findall(r"[\w€£$]+", self.text))
        self.raw_words = re.findall(r"[\w]+", text)

    def is_question(self) -> bool:
        first = next(iter(re.findall(r"\w+", self.text)), "")
        return "?" in self.text or first in QUESTION_WORDS

    def has(self, *phrases: str) -> bool:
        return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", self.text) for p in phrases)

    def contains(self, *pieces: str) -> bool:
        return any(p in self.text for p in pieces)

    def mentions_amount(self, value: float) -> bool:
        # same as mentions_number, except "4-star" or "10 days" isn't money
        n = int(value) if float(value).is_integer() else value
        for match in re.finditer(rf"(?<![\w.]){re.escape(str(n))}(?![\w.])", self.text):
            if not re.match(rf"\s*-?\s*({NOT_MONEY})\b", self.text[match.end():]):
                return True
        thousands = f"{value / 1000:g}"
        return self.has(f"{thousands}k", f"{thousands} k", f"{thousands} grand", f"{thousands} thousand")

    def mentions_number(self, value: float) -> bool:
        n = int(value) if float(value).is_integer() else value
        if str(n) in self.words or any(NUMBER_WORDS.get(w) == n for w in self.words):
            return True
        thousands = f"{value / 1000:g}"
        return self.has(f"{thousands}k", f"{thousands} k", f"{thousands} grand", f"{thousands} thousand")


def _country_mentioned(country: str, msg: Message) -> bool:
    words = msg.raw_words
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            chunk = " ".join(words[i : i + size])
            # codes only in capitals, or "in"/"to"/"me" match India/Tonga/Montenegro
            if chunk.lower() not in COUNTRY_ALIASES and len(chunk) <= 3 and not chunk.isupper():
                continue
            try:
                if normalize_country(chunk) == country:
                    return True
            except ValueError:
                pass
    return False


def date_mentioned(value: date, msg: Message) -> bool:
    month = value.strftime("%B").lower()
    return (
        msg.has(month, month[:3])
        or str(value.day) in msg.words
        or bool(msg.words & RELATIVE_DATE_WORDS)
    )


def is_grounded(field: str, value, msg: Message, region_kept: bool = False) -> bool:
    if field == "country":
        try:
            country = normalize_country(value)
        except ValueError:
            return True  # let validation reject it with a proper message
        return region_kept or _country_mentioned(country, msg)
    if field == "region":
        return not is_whole_message(value, msg) and all(
            w in msg.words for w in re.findall(r"\w+", value.lower())
        )
    if field == "party_size":
        return msg.mentions_number(value) or bool(msg.words & PEOPLE_WORDS)
    if field == "num_days":
        return (
            msg.mentions_number(value)
            or (value % 7 == 0 and msg.contains("week"))
            or (value == 14 and msg.has("fortnight"))
            or (value in (2, 3) and msg.has("weekend"))
        )
    if field == "budget_amount":
        return msg.mentions_amount(value)
    if field == "budget_currency":
        code = str(value).strip().upper()
        return code.lower() in msg.words or any(
            msg.has(word) for word, c in CURRENCY_WORDS.items() if c == code
        )
    if field == "budget_scope":
        return msg.has(*PER_PERSON_PHRASES) if value == "per_person" else msg.has(*TOTAL_PHRASES)
    if field == "hotel_stars":
        return msg.contains("star") and msg.mentions_number(value)
    if field == "min_review_score":
        return msg.contains(*REVIEW_WORDS) and msg.mentions_number(value)
    return True


def interest_mentioned(interest: str, msg: Message) -> bool:
    return bool(msg.words & INTEREST_WORDS[interest])


YES_REPLIES = {
    "yes", "y", "yeah", "yep", "yup", "ya", "ok", "okay", "k", "sure", "correct", "right",
    "exactly", "perfect", "confirm", "great", "yes please", "looks good", "sounds good",
    "all good", "that's right", "thats right", "go ahead", "that is right",
}
NO_REPLIES = {"no", "n", "nope", "nah", "wrong", "incorrect", "not right", "that's wrong"}
NOTHING_REPLIES = {"nothing", "nth", "none", "nvm", "never mind", "no changes", "nothing to change"}
NO_STARTS = {"no", "nope", "nah", "wrong"}


def correct_message_type(model_type: str, message: str) -> str:
    text = " ".join(re.findall(r"[\w']+", message.lower()))
    if text in YES_REPLIES:
        return "yes"
    if text in NOTHING_REPLIES:
        return "trip_info"
    if text in NO_REPLIES:
        return "no"
    if text.split(" ")[0] in NO_STARTS and model_type in ("trip_info", "greeting", "yes"):
        return "no"
    # the model calls slang like "nth" non-English; only override that for short text
    if model_type == "non_english" and message.isascii() and len(text.split()) <= 3:
        return "off_topic"
    return model_type


MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"}
UNCLEAR_HINTS = {
    "budget_amount": {"cheap", "budget", "money", "expensive", "afford", "affordable", "luxury",
                      "price", "cost", "spend", "pricey", "dollars", "pounds", "euros", "$", "£"},
    "budget_currency": {"$", "£", "dollar", "dollars", "pound", "pounds", "currency", "bucks"},
    "start_date": MONTHS | {m[:3] for m in MONTHS} | {
        "summer", "winter", "spring", "autumn", "fall", "soon", "next", "holiday", "holidays",
        "christmas", "easter", "weekend", "month", "year", "decamber"},
    "num_days": {"few", "couple", "several", "week", "weeks", "while", "days", "nights"},
    "party_size": PEOPLE_WORDS | {"few", "some", "us", "we", "group", "couple"},
    "country": {"somewhere", "anywhere", "europe", "asia", "africa", "america", "abroad"},
}


def unclear_mentioned(field: str, msg: Message) -> bool:
    hints = UNCLEAR_HINTS.get(field)
    return hints is None or bool(msg.words & hints)


def tidy_extraction(extraction, message: str):
    msg = Message(message)
    return extraction.model_copy(update={
        "message_type": correct_message_type(extraction.message_type, message),
        "unclear": [f for f in extraction.unclear if unclear_mentioned(f, msg)],
    })


def question_mentioned(question: str, msg: Message) -> bool:
    if not msg.is_question():
        return False
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2} - STOP_WORDS
    return bool(words & msg.words)


def is_whole_message(value: str, msg: Message) -> bool:
    words = re.findall(r"\w+", value.lower())
    return len(words) >= 3 and words == re.findall(r"\w+", msg.text)


def either_or(value: float, msg: Message) -> list[str] | None:
    n = str(int(value) if float(value).is_integer() else value)
    for a, b in re.findall(r"(\d[\d.]*k?)\s*(?:or|/)\s*(\d[\d.]*k?)", msg.text):
        if n in (a, b):
            return [a, b]
    return None
