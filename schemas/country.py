import pycountry

# names pycountry doesn't know. "Korea" is left out on purpose so the bot asks.
COUNTRY_ALIASES: dict[str, str] = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "america": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "uae": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "russia": "Russian Federation",
    "turkey": "Türkiye",
    "turkiye": "Türkiye",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "italia": "Italy",
    "espana": "Spain",
    "españa": "Spain",
    "deutschland": "Germany",
    "palestine": "Palestine, State of",
    "ivory coast": "Côte d'Ivoire",
    "cape verde": "Cabo Verde",
    "macedonia": "North Macedonia",
    "swaziland": "Eswatini",
    "burma": "Myanmar",
    "vatican": "Holy See (Vatican City State)",
    "vatican city": "Holy See (Vatican City State)",
    "brunei": "Brunei Darussalam",
    "micronesia": "Micronesia, Federated States of",
}

DISPLAY_NAMES: dict[str, str] = {
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "Palestine, State of": "Palestine",
    "Holy See (Vatican City State)": "Vatican City",
    "Brunei Darussalam": "Brunei",
    "Micronesia, Federated States of": "Micronesia",
}


def normalize_country(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Country cannot be empty.")
    lookup_name = COUNTRY_ALIASES.get(text.lower(), text)
    try:
        country = pycountry.countries.lookup(lookup_name)
    except LookupError:
        raise ValueError(
            f"I don't recognise '{text}' as a country. Could you write the country's full name?"
        ) from None
    name = getattr(country, "common_name", None) or country.name
    return DISPLAY_NAMES.get(name, name)
