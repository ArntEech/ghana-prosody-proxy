import re
import unicodedata


def normalize_unicode(text):
    """Normalize Unicode characters."""
    return unicodedata.normalize("NFKC", str(text))


def replace_urls(text):
    """Replace URLs with a standard token."""
    return re.sub(
        r"https?://\S+|www\.\S+",
        "<URL>",
        text
    )


def replace_mentions(text):
    """Replace Twitter/X mentions with a standard token."""
    return re.sub(
        r"@\w+",
        "<USER>",
        text
    )


def normalize_whitespace(text):
    """Remove excessive whitespace."""
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def preprocess_text(text):
    """
    Perform basic preprocessing while preserving
    expressive information such as emojis, capitalization,
    punctuation, and repeated characters.
    """

    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = replace_urls(text)
    text = replace_mentions(text)
    text = normalize_whitespace(text)

    return text