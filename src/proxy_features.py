import re
from collections import Counter


# ============================================================
# GHANAIAN PRAGMATIC PARTICLES AND INTERJECTIONS
# ============================================================

PRAGMATIC_PARTICLES = {
    "o",
    "oo",
    "ooo",
    "oooo",
    "paa",
    "saa",
    "saaa",
    "aa",
    "aaa",
    "eh",
    "ehe",
    "waa",
    "waaa",
    "yoo",
}


INTERJECTIONS = {
    "ei",
    "eii",
    "eiii",
    "eiiii",
    "eish",
    "eh",
    "ehe",
    "hmm",
    "hmmm",
    "chai",
    "charley",
    "chale",
    "ah",
    "aha",
    "oh",
    "wow",
}


# ============================================================
# EMOJI DETECTION
# ============================================================

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


# ============================================================
# CHARACTER ELONGATION
# ============================================================

def find_elongations(text):
    """
    Detect sequences where the same character is repeated
    three or more times.

    Example:
    chaleeee -> e repeated 4 times
    saaaa -> a repeated 4 times
    """

    pattern = r"(.)\1{2,}"

    matches = list(
        re.finditer(pattern, text, flags=re.IGNORECASE)
    )

    return matches


def extract_elongation_features(text):

    matches = find_elongations(text)

    repetition_lengths = [
        len(match.group())
        for match in matches
    ]

    return {
        "elongation_count": len(matches),

        "max_repetition_length":
            max(repetition_lengths)
            if repetition_lengths else 0,

        "has_elongation":
            int(len(matches) > 0),
    }


# ============================================================
# EMOJI FEATURES
# ============================================================

def extract_emoji_features(text):

    emojis = EMOJI_PATTERN.findall(text)

    emoji_string = "".join(emojis)

    emoji_count = len(emoji_string)

    return {
        "emoji_count": emoji_count,

        "has_emoji":
            int(emoji_count > 0),

        "emoji_characters":
            emoji_string,
    }


# ============================================================
# PUNCTUATION FEATURES
# ============================================================

def extract_punctuation_features(text):

    exclamation_count = text.count("!")
    question_count = text.count("?")

    repeated_exclamation = int(
        bool(re.search(r"!{2,}", text))
    )

    repeated_question = int(
        bool(re.search(r"\?{2,}", text))
    )

    mixed_punctuation = int(
        bool(
            re.search(
                r"(\?!|!\?|\?\!|\!\?)",
                text
            )
        )
    )

    ellipsis = int(
        bool(re.search(r"\.{3,}", text))
    )

    punctuation_intensity = (
        exclamation_count
        + question_count
    )

    return {
        "exclamation_count":
            exclamation_count,

        "question_count":
            question_count,

        "repeated_exclamation":
            repeated_exclamation,

        "repeated_question":
            repeated_question,

        "mixed_punctuation":
            mixed_punctuation,

        "ellipsis":
            ellipsis,

        "punctuation_intensity":
            punctuation_intensity,
    }


# ============================================================
# CAPITALISATION FEATURES
# ============================================================

def extract_capitalisation_features(text):

    letters = [
        char for char in text
        if char.isalpha()
    ]

    uppercase_letters = [
        char for char in letters
        if char.isupper()
    ]

    uppercase_ratio = (
        len(uppercase_letters) / len(letters)
        if letters else 0
    )

    tokens = re.findall(
        r"\b[A-Za-z]+\b",
        text
    )

    full_caps_tokens = [
        token
        for token in tokens
        if len(token) > 1
        and token.isupper()
    ]

    return {
        "uppercase_ratio":
            round(uppercase_ratio, 4),

        "full_caps_token_count":
            len(full_caps_tokens),

        "has_full_caps":
            int(len(full_caps_tokens) > 0),
    }


# ============================================================
# PRAGMATIC PARTICLES
# ============================================================

def extract_particle_features(text):

    tokens = re.findall(
        r"\b[\w']+\b",
        text.lower()
    )

    found_particles = [
        token
        for token in tokens
        if token in PRAGMATIC_PARTICLES
    ]

    final_particle = ""

    if tokens:
        last_token = tokens[-1]

        if last_token in PRAGMATIC_PARTICLES:
            final_particle = last_token

    return {
        "particle_count":
            len(found_particles),

        "has_particle":
            int(len(found_particles) > 0),

        "final_particle":
            final_particle,

        "has_final_particle":
            int(final_particle != ""),
    }


# ============================================================
# INTERJECTIONS
# ============================================================

def extract_interjection_features(text):

    tokens = re.findall(
        r"\b[\w']+\b",
        text.lower()
    )

    found_interjections = [
        token
        for token in tokens
        if token in INTERJECTIONS
    ]

    return {
        "interjection_count":
            len(found_interjections),

        "has_interjection":
            int(len(found_interjections) > 0),

        "interjections_found":
            ",".join(
                sorted(
                    set(found_interjections)
                )
            ),
    }


# ============================================================
# TEXT LENGTH FEATURES
# ============================================================

def extract_text_features(text):

    tokens = re.findall(
        r"\b[\w']+\b",
        text
    )

    return {
        "character_count":
            len(text),

        "word_count":
            len(tokens),
    }


# ============================================================
# MAIN FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_proxy_features(text):
    """
    Extract all proposed prosodic and pragmatic
    proxy features from one social-media post.
    """

    if not isinstance(text, str):
        text = str(text)

    features = {}

    features.update(
        extract_elongation_features(text)
    )

    features.update(
        extract_emoji_features(text)
    )

    features.update(
        extract_punctuation_features(text)
    )

    features.update(
        extract_capitalisation_features(text)
    )

    features.update(
        extract_particle_features(text)
    )

    features.update(
        extract_interjection_features(text)
    )

    features.update(
        extract_text_features(text)
    )

    # ========================================================
    # TOTAL PROXY COUNT
    # ========================================================

    proxy_count = (
        features["has_elongation"]
        + features["has_emoji"]
        + features["has_particle"]
        + features["has_interjection"]
        + features["repeated_exclamation"]
        + features["repeated_question"]
        + features["mixed_punctuation"]
        + features["has_full_caps"]
    )

    features["proxy_count"] = proxy_count

    return features