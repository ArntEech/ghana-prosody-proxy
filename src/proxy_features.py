import re
from collections import Counter

try:
    import emoji
except ImportError:
    emoji = None


FINAL_PARTICLES = {
    "o", "oo", "ooo", "oooo",
    "aa", "aaa",
    "paa", "pa",
    "saa", "saaa",
    "eh", "ehe"
}

INTERJECTIONS = {
    "ei", "eii", "eiii",
    "eish", "eiiish",
    "eh", "ehe",
    "hmm", "hmmm",
    "chai", "chale"
}


def get_elongation_features(text):
    """Detect deliberate repeated characters."""

    tokens = re.findall(r"\b\w+\b", text.lower())

    elongated = []

    for token in tokens:
        matches = re.findall(r"(.)\1{2,}", token)

        if matches:
            elongated.append(token)

    max_repeat = 0

    for match in re.finditer(r"(.)\1+", text.lower()):
        length = len(match.group(0))
        max_repeat = max(max_repeat, length)

    token_count = max(len(tokens), 1)

    return {
        "elongated_token_count": len(elongated),
        "max_character_repetition": max_repeat,
        "elongation_index": len(elongated) / token_count,
    }


def get_punctuation_features(text):

    punctuation_stacks = re.findall(r"[!?]{2,}", text)

    return {
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "ellipsis_count": len(re.findall(r"\.{3,}", text)),
        "punctuation_stack_count": len(punctuation_stacks),
        "max_punctuation_stack": max(
            [len(x) for x in punctuation_stacks],
            default=0
        )
    }


def get_capitalization_features(text):

    tokens = re.findall(r"\b[A-Za-z]+\b", text)

    alpha_chars = [c for c in text if c.isalpha()]

    uppercase_chars = [c for c in alpha_chars if c.isupper()]

    uppercase_ratio = (
        len(uppercase_chars) / len(alpha_chars)
        if alpha_chars else 0
    )

    all_caps_tokens = [
        token for token in tokens
        if len(token) > 1 and token.isupper()
    ]

    return {
        "uppercase_ratio": uppercase_ratio,
        "all_caps_token_count": len(all_caps_tokens)
    }


def get_emoji_features(text):

    if emoji is None:
        return {
            "emoji_count": 0,
            "emoji_present": 0
        }

    emojis = emoji.emoji_list(text)

    return {
        "emoji_count": len(emojis),
        "emoji_present": int(len(emojis) > 0)
    }


def get_pragmatic_features(text):

    tokens = re.findall(r"\b[\w']+\b", text.lower())

    if not tokens:
        return {
            "final_particle_present": 0,
            "final_particle_type": "none",
            "interjection_count": 0,
            "interjection_present": 0
        }

    final_token = tokens[-1]

    final_particle_present = int(final_token in FINAL_PARTICLES)

    interjections = [
        token for token in tokens
        if token in INTERJECTIONS
    ]

    return {
        "final_particle_present": final_particle_present,
        "final_particle_type": (
            final_token
            if final_particle_present
            else "none"
        ),
        "interjection_count": len(interjections),
        "interjection_present": int(len(interjections) > 0)
    }


def extract_proxy_features(text):
    """Extract all textual prosodic/pragmatic proxy features."""

    if not isinstance(text, str):
        text = ""

    features = {}

    features.update(get_elongation_features(text))
    features.update(get_punctuation_features(text))
    features.update(get_capitalization_features(text))
    features.update(get_emoji_features(text))
    features.update(get_pragmatic_features(text))

    features["text_length"] = len(text)
    features["word_count"] = len(
        re.findall(r"\b\w+\b", text)
    )

    return features