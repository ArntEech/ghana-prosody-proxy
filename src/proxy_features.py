import re
import unicodedata


# --------------------------------------------------
# Ghanaian pragmatic markers / particles
# --------------------------------------------------

FINAL_PARTICLES = [
    "o",
    "oo",
    "ooo",
    "aa",
    "saa",
    "paa",
    "saa",
    "waa",
    "wai",
    "wae",
    "la",
    "koraa",
    "kraa",
]


# --------------------------------------------------
# Ghanaian interjections
# --------------------------------------------------

INTERJECTIONS = [
    "ei",
    "eii",
    "ehe",
    "eh",
    "eish",
    "hmm",
    "hmmm",
    "huh",
]


def count_repeated_characters(text):
    """
    Counts occurrences of characters repeated
    three or more times.

    Example:
        Chaleeee -> 1
        soooo -> 1
    """

    matches = re.findall(
        r"(.)\1{2,}",
        text.lower()
    )

    return len(matches)


def max_character_elongation(text):
    """
    Returns the longest sequence of repeated
    characters.
    """

    matches = re.findall(
        r"(.)\1+",
        text.lower()
    )

    if not matches:
        return 1

    max_length = 1

    for match in re.finditer(
        r"(.)\1+",
        text.lower()
    ):
        max_length = max(
            max_length,
            len(match.group())
        )

    return max_length


def count_exclamation_marks(text):
    return text.count("!")


def count_question_marks(text):
    return text.count("?")


def count_periods(text):
    return text.count(".")


def count_ellipsis(text):
    return len(
        re.findall(
            r"\.{3,}",
            text
        )
    )


def count_punctuation_stacks(text):
    """
    Counts combinations such as:

        !!!
        ???
        ?!
        !?
    """

    return len(
        re.findall(
            r"[!?]{2,}",
            text
        )
    )


def uppercase_ratio(text):
    """
    Ratio of uppercase alphabetic characters
    to all alphabetic characters.
    """

    letters = [
        char for char in text
        if char.isalpha()
    ]

    if not letters:
        return 0.0

    uppercase = [
        char for char in letters
        if char.isupper()
    ]

    return len(uppercase) / len(letters)


def count_emojis(text):
    """
    Approximate emoji count using Unicode ranges.
    """

    count = 0

    for char in text:

        codepoint = ord(char)

        if (
            0x1F300 <= codepoint <= 0x1FAFF
            or 0x2600 <= codepoint <= 0x27BF
        ):
            count += 1

    return count


def has_emoji(text):
    return int(count_emojis(text) > 0)


def count_final_particles(text):
    """
    Detects Ghanaian pragmatic particles,
    especially near the end of an utterance.
    """

    tokens = re.findall(
        r"\b[\wÀ-ÿ]+\b",
        text.lower()
    )

    if not tokens:
        return 0

    count = 0

    # Examine the final five tokens
    final_tokens = tokens[-5:]

    for token in final_tokens:

        if token in FINAL_PARTICLES:
            count += 1

    return count


def has_final_particle(text):
    return int(
        count_final_particles(text) > 0
    )


def count_interjections(text):

    tokens = re.findall(
        r"\b[\wÀ-ÿ]+\b",
        text.lower()
    )

    return sum(
        token in INTERJECTIONS
        for token in tokens
    )


def has_interjection(text):
    return int(
        count_interjections(text) > 0
    )


def token_count(text):

    return len(
        re.findall(
            r"\b[\wÀ-ÿ]+\b",
            text
        )
    )


def repeated_punctuation_ratio(text):

    if not text:
        return 0.0

    repeated = len(
        re.findall(
            r"[!?.,]{2,}",
            text
        )
    )

    return repeated / max(
        len(text),
        1
    )


def elongation_ratio(text):

    if not text:
        return 0.0

    repeated_chars = sum(
        len(match.group()) - 1
        for match in re.finditer(
            r"(.)\1+",
            text.lower()
        )
    )

    return repeated_chars / max(
        len(text),
        1
    )


def extract_proxy_features(text):

    text = str(text)

    features = {

        # ------------------------------------------
        # Elongation
        # ------------------------------------------

        "repeated_char_count":
            count_repeated_characters(text),

        "max_elongation":
            max_character_elongation(text),

        "elongation_ratio":
            elongation_ratio(text),

        # ------------------------------------------
        # Punctuation
        # ------------------------------------------

        "exclamation_count":
            count_exclamation_marks(text),

        "question_count":
            count_question_marks(text),

        "period_count":
            count_periods(text),

        "ellipsis_count":
            count_ellipsis(text),

        "punctuation_stack_count":
            count_punctuation_stacks(text),

        "repeated_punctuation_ratio":
            repeated_punctuation_ratio(text),

        # ------------------------------------------
        # Capitalization
        # ------------------------------------------

        "uppercase_ratio":
            uppercase_ratio(text),

        # ------------------------------------------
        # Emojis
        # ------------------------------------------

        "emoji_count":
            count_emojis(text),

        "has_emoji":
            has_emoji(text),

        # ------------------------------------------
        # Final particles
        # ------------------------------------------

        "final_particle_count":
            count_final_particles(text),

        "has_final_particle":
            has_final_particle(text),

        # ------------------------------------------
        # Interjections
        # ------------------------------------------

        "interjection_count":
            count_interjections(text),

        "has_interjection":
            has_interjection(text),

        # ------------------------------------------
        # Basic textual intensity
        # ------------------------------------------

        "text_length":
            len(text),

        "token_count":
            token_count(text),

        "exclamation_question_count":
            count_exclamation_marks(text)
            + count_question_marks(text),

        "emoji_punctuation_interaction":
            count_emojis(text)
            * (
                1
                + count_punctuation_stacks(text)
            ),

        "elongation_punctuation_interaction":
            count_repeated_characters(text)
            * (
                1
                + count_punctuation_stacks(text)
            ),
    }

    return features