import re


GHANAIAN_PIDGIN_MARKERS = {
    "dey", "chale", "charley", "abi",
    "waa", "paa", "make", "go",
    "no", "dem", "am", "small"
}

AKAN_MARKERS = {
    "medaase", "akwaaba", "yɛ",
    "yɛn", "paa", "oo", "ei"
}


def detect_language_features(text):

    tokens = set(
        re.findall(r"\b[\w']+\b", text.lower())
    )

    pidgin_matches = tokens.intersection(
        GHANAIAN_PIDGIN_MARKERS
    )

    akan_matches = tokens.intersection(
        AKAN_MARKERS
    )

    marker_count = (
        len(pidgin_matches) +
        len(akan_matches)
    )

    language_mix_score = (
        marker_count / max(len(tokens), 1)
    )

    return {
        "pidgin_marker_count": len(pidgin_matches),
        "akan_marker_count": len(akan_matches),
        "language_mix_score": language_mix_score,
        "language_mix_present": int(marker_count > 0)
    }