import re


# Candidate Ghanaian / Ghanaian-English / Pidgin pragmatic markers
PRAGMATIC_PARTICLES = {
    "o",
    "oo",
    "ooo",
    "paa",
    "saaa",
    "aa",
    "eh",
    "ehe",
    "ei",
    "eish",
    "hmm",
    "hmmm"
}


def detect_elongation(text):
    """
    Detect repeated characters such as:
    chaleeee
    soooo
    whaaaat
    """

    pattern = r"(\w)\1{2,}"

    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    elongated_sequences = re.findall(
        r"\w*(\w)\1{2,}\w*",
        text,
        flags=re.IGNORECASE
    )

    return {
        "elongation_count": len(matches),
        "elongated_tokens": len(elongated_sequences)
    }


def max_character_repetition(text):
    """
    Find the longest consecutive repeated-character sequence.
    """

    matches = re.findall(r"(.)\1+", text)

    if not matches:
        return 1

    sequences = re.findall(r"(.)\1+", text)

    max_length = 1

    for char in set(sequences):
        pattern = re.escape(char) + "+"
        runs = re.findall(pattern, text)

        for run in runs:
            max_length = max(max_length, len(run))

    return max_length


def calculate_elongation_index(text):
    """
    Calculate a simple normalized elongation measure.

    EI = number of repeated-character sequences / total tokens
    """

    tokens = text.split()

    if len(tokens) == 0:
        return 0.0

    repeated_sequences = re.findall(
        r"(\w)\1{2,}",
        text,
        flags=re.IGNORECASE
    )

    return len(repeated_sequences) / len(tokens)


def punctuation_features(text):
    """
    Extract punctuation-based expressive features.
    """

    return {
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "combined_punctuation": int("?!" in text or "!?" in text),
        "repeated_exclamation": int("!!" in text),
        "repeated_question": int("??" in text),
        "ellipsis_count": text.count("...")
    }


def capitalization_features(text):
    """
    Extract capitalization features.
    """

    words = re.findall(r"\b[A-Za-z]+\b", text)

    if not words:
        return {
            "uppercase_ratio": 0.0,
            "full_caps_tokens": 0
        }

    uppercase_words = [
        word for word in words
        if len(word) > 1 and word.isupper()
    ]

    uppercase_ratio = len(uppercase_words) / len(words)

    return {
        "uppercase_ratio": uppercase_ratio,
        "full_caps_tokens": len(uppercase_words)
    }


def emoji_features(text):
    """
    Detect approximate emoji usage using Unicode ranges.
    """

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )

    emojis = emoji_pattern.findall(text)

    emoji_count = sum(len(group) for group in emojis)

    return {
        "emoji_present": int(emoji_count > 0),
        "emoji_count": emoji_count
    }


def pragmatic_particle_features(text):
    """
    Detect candidate pragmatic particles and interjections.
    """

    tokens = re.findall(
        r"\b[\w']+\b",
        text.lower()
    )

    found_particles = [
        token for token in tokens
        if token in PRAGMATIC_PARTICLES
    ]

    final_particle = None

    if tokens and tokens[-1] in PRAGMATIC_PARTICLES:
        final_particle = tokens[-1]

    return {
        "particle_count": len(found_particles),
        "final_particle_present": int(final_particle is not None),
        "final_particle": final_particle if final_particle else "none"
    }


def basic_text_features(text):
    """
    Extract general text statistics.
    """

    tokens = text.split()

    return {
        "word_count": len(tokens),
        "character_count": len(text)
    }


def extract_proxy_features(text):
    """
    Extract all prosodic and pragmatic proxy features.
    """

    features = {}

    features.update(detect_elongation(text))
    features["max_character_repetition"] = (
        max_character_repetition(text)
    )

    features["elongation_index"] = (
        calculate_elongation_index(text)
    )

    features.update(punctuation_features(text))
    features.update(capitalization_features(text))
    features.update(emoji_features(text))
    features.update(pragmatic_particle_features(text))
    features.update(basic_text_features(text))

    return features