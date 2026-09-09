"""Conservative, explicit aliases. Original annotations are never rewritten."""

ALIASES = {
    "STEMI": "STEMI-ACS",
    "COPD Asthma": "Asthma-COPD",
    "Pituitary Macroadenoma": "Pituitary Macroadenomas",
    "Pituitary Microadenoma": "Pituitary Microadenomas",
}


def normalize_diagnosis(text: str) -> str:
    text = " ".join(text.split())
    return ALIASES.get(text, text).casefold()
