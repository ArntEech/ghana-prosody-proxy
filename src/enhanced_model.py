from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack


PROXY_COLUMNS = [
    "elongated_token_count",
    "max_character_repetition",
    "elongation_index",
    "exclamation_count",
    "question_count",
    "ellipsis_count",
    "punctuation_stack_count",
    "max_punctuation_stack",
    "uppercase_ratio",
    "all_caps_token_count",
    "emoji_count",
    "emoji_present",
    "final_particle_present",
    "interjection_count",
    "interjection_present",
    "text_length",
    "word_count",
    "pidgin_marker_count",
    "akan_marker_count",
    "language_mix_score",
    "language_mix_present"
]


def build_proxy_model():

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    return vectorizer, classifier


def combine_features(text_matrix, proxy_matrix):

    return hstack([
        text_matrix,
        proxy_matrix
    ])