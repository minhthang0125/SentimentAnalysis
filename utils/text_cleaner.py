import re
from urllib.parse import urlparse


WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_CHARS_PATTERN = re.compile(r"[\r\n\t]+")
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s,.!?]")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

YOUTUBE_DOMAINS = ("youtube.com", "youtu.be")


def parse_url(value: str):
    try:
        return urlparse(value)
    except ValueError:
        return None


def is_valid_url(value: str) -> bool:
    parsed = parse_url(value)
    return parsed is not None and parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_youtube_url(value: str) -> bool:
    if not is_valid_url(value):
        return False

    hostname = (parse_url(value).netloc or "").lower().split(":", 1)[0]
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in YOUTUBE_DOMAINS)


def clean_comment(text: str, remove_emojis: bool = True) -> str:
    if not text:
        return ""

    cleaned = CONTROL_CHARS_PATTERN.sub(" ", text)
    if remove_emojis:
        cleaned = EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = SPECIAL_CHARS_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned
