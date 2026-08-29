from pathlib import Path
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from scipy.sparse import csr_matrix, hstack


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/annotated/prosody_proxy_annotations.csv"
OUTPUT_FILE = ROOT / "reports/ablation_results.csv"


FEATURE_GROUPS = {

    "Baseline": [],

    "Elongation": [
        "elongated_token_count",
        "max_character_repetition",
        "elongation_index"
    ],

    "Emoji": [
        "emoji_count",
        "emoji_present"
    ],

    "Punctuation_Capitalisation": [
        "exclamation_count",
        "question_count",
        "ellipsis_count",
        "punctuation_stack_count",
        "max_punctuation_stack",
        "uppercase_ratio",
        "all_caps_token_count"
    ],

    "Pragmatic": [
        "final_particle_present",
        "interjection_count",
        "interjection_present"
    ],

    "Language_Mixing": [
        "pidgin_marker_count",
        "akan_marker_count",
        "language_mix_score",
        "language_mix_present"
    ]
}


def evaluate(feature_columns, train_df, test_df):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    train_text = vectorizer.fit_transform(
        train_df["text"]
    )

    test_text = vectorizer.transform(
        test_df["text"]
    )

    if feature_columns:

        train_proxy = csr_matrix(
            train_df[feature_columns]
            .fillna(0)
            .astype(float)
            .values
        )

        test_proxy = csr_matrix(
            test_df[feature_columns]
            .fillna(0)
            .astype(float)
            .values
        )

        X_train = hstack(
            [train_text, train_proxy]
        )

        X_test = hstack(
            [test_text, test_proxy]
        )

    else:

        X_train = train_text
        X_test = test_text

    model.fit(
        X_train,
        train_df["label"]
    )

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(
            test_df["label"],
            predictions
        ),

        "macro_f1": f1_score(
            test_df["label"],
            predictions,
            average="macro"
        ),

        "weighted_f1": f1_score(
            test_df["label"],
            predictions,
            average="weighted"
        )
    }


def main():

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(
        subset=["text", "label"]
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    results = []

    for name, columns in FEATURE_GROUPS.items():

        metrics = evaluate(
            columns,
            train_df,
            test_df
        )

        metrics["experiment"] = name

        results.append(metrics)

    all_features = []

    for columns in FEATURE_GROUPS.values():
        all_features.extend(columns)

    metrics = evaluate(
        all_features,
        train_df,
        test_df
    )

    metrics["experiment"] = "Full_Model"

    results.append(metrics)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(result_df)


if __name__ == "__main__":
    main()