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

SUPPORTED_PLATFORMS = {
    "youtube": ("youtube.com", "youtu.be"),
}


def parse_url(value: str):
    try:
        return urlparse(value)
    except ValueError:
        return None


def is_valid_url(value: str) -> bool:
    parsed = parse_url(value)
    return parsed is not None and parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def detect_supported_platform(value: str) -> str | None:
    if not is_valid_url(value):
        return None

    hostname = (parse_url(value).netloc or "").lower()
    if ":" in hostname:
        hostname = hostname.split(":", 1)[0]

    for platform_name, supported_domains in SUPPORTED_PLATFORMS.items():
        if any(
            hostname == supported_domain or hostname.endswith(f".{supported_domain}")
            for supported_domain in supported_domains
        ):
            return platform_name
    return None


def is_supported_platform_url(value: str) -> bool:
    return detect_supported_platform(value) is not None


def clean_comment(text: str, remove_emojis: bool = True) -> str:
    if not text:
        return ""

    cleaned = CONTROL_CHARS_PATTERN.sub(" ", text)
    if remove_emojis:
        cleaned = EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = SPECIAL_CHARS_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned
