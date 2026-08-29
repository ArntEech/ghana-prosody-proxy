import re
import unicodedata


def preprocess_text(text):
    """Clean technical noise while preserving expressive writing."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Replace URLs and mentions.
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)

    # Preserve hashtag words but remove #.
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize whitespace only.
    text = re.sub(r"\s+", " ", text).strip()

    return text