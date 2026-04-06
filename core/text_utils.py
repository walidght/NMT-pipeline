import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\u200b\u200e\u200f\ufeff]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
