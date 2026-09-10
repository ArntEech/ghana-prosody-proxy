from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "corpus_with_features.csv"
OUTPUT_FILE = ROOT / "reports" / "ablation_results.csv"


# ---------------------------------------------------------
# Proxy feature groups
# ---------------------------------------------------------

FEATURE_GROUPS = {
    "Elongation": [
        "repeated_char_count",
        "max_elongation",
        "elongation_ratio",
    ],

    "Emoji": [
        "emoji_count",
        "has_emoji",
    ],

    "Punctuation_Capitalisation": [
        "exclamation_count",
        "question_count",
        "period_count",
        "ellipsis_count",
        "punctuation_stack_count",
        "repeated_punctuation_ratio",
        "uppercase_ratio",
    ],

    "Pragmatic": [
        "final_particle_count",
        "has_final_particle",
        "interjection_count",
        "has_interjection",
    ],

    "Language_Mixing": [
        # These are not currently present in the generated
        # feature file, so this group will be skipped unless
        # the columns exist.
        "pidgin_marker_count",
        "akan_marker_count",
        "language_mix_score",
        "language_mix_present",
    ],
}


ALL_PROXY_FEATURES = [
    "repeated_char_count",
    "max_elongation",
    "elongation_ratio",
    "exclamation_count",
    "question_count",
    "period_count",
    "ellipsis_count",
    "punctuation_stack_count",
    "repeated_punctuation_ratio",
    "uppercase_ratio",
    "emoji_count",
    "has_emoji",
    "final_particle_count",
    "has_final_particle",
    "interjection_count",
    "has_interjection",
    "text_length",
    "token_count",
    "exclamation_question_count",
    "emoji_punctuation_interaction",
    "elongation_punctuation_interaction",
]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def train_and_evaluate(
    X_train_text,
    X_test_text,
    y_train,
    y_test,
    train_proxy=None,
    test_proxy=None,
):
    """
    Train TF-IDF + Logistic Regression.
    Optionally append proxy features.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=10000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    if train_proxy is not None and test_proxy is not None:
        from scipy.sparse import csr_matrix, hstack

        scaler = StandardScaler()

        train_proxy_scaled = scaler.fit_transform(train_proxy)
        test_proxy_scaled = scaler.transform(test_proxy)

        train_proxy_sparse = csr_matrix(train_proxy_scaled)
        test_proxy_sparse = csr_matrix(test_proxy_scaled)
        
        X_train = hstack(
            [X_train_tfidf, train_proxy_sparse],
            format="csr",
        )

        X_test = hstack(
            [X_test_tfidf, test_proxy_sparse],
            format="csr",
        )
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf

    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )
    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "tfidf_feature_count": len(vectorizer.vocabulary_),
    }


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    # -----------------------------------------------------
    # Load the SAME dataset used by the baseline/proxy model
    # -----------------------------------------------------

    print(f"\nLoading dataset:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape}")

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    required_columns = [
        "clean_text",
        "label",
        "split",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # -----------------------------------------------------
    # Use the EXISTING train/test split
    # -----------------------------------------------------

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"\nTraining records: {len(train_df)}")
    print(f"Test records:     {len(test_df)}")

    # -----------------------------------------------------
    # Prepare text and labels
    # -----------------------------------------------------

    X_train_text = train_df["clean_text"].fillna("")
    X_test_text = test_df["clean_text"].fillna("")

    y_train = train_df["label"]
    y_test = test_df["label"]

    # -----------------------------------------------------
    # Check available proxy features
    # -----------------------------------------------------

    available_proxy_features = [
        col
        for col in ALL_PROXY_FEATURES
        if col in df.columns
    ]

    print(
        f"\nAvailable proxy features: "
        f"{len(available_proxy_features)}"
    )

    # -----------------------------------------------------
    # Results container
    # -----------------------------------------------------

    results = []

    # -----------------------------------------------------
    # 1. Baseline
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("Experiment: Baseline")
    print("-" * 70)

    baseline = train_and_evaluate(
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    )

    baseline["experiment"] = "Baseline"
    baseline["proxy_feature_count"] = 0
    baseline["total_feature_count"] = (
        baseline["tfidf_feature_count"]
    )

    results.append(baseline)

    print(
        f"Accuracy:  {baseline['accuracy']:.4f}"
    )
    print(
        f"Macro-F1:  {baseline['macro_f1']:.4f}"
    )
    print(
        f"Weighted-F1:{baseline['weighted_f1']:.4f}"
    )

    # -----------------------------------------------------
    # 2. Individual proxy groups
    # -----------------------------------------------------

    for experiment_name, requested_features in FEATURE_GROUPS.items():

        usable_features = [
            col
            for col in requested_features
            if col in df.columns
        ]

        # Skip unavailable groups
        if not usable_features:
            print(
                f"\nSkipping {experiment_name}: "
                f"no matching columns found."
            )
            continue

        print("\n" + "-" * 70)
        print(f"Experiment: {experiment_name}")
        print("-" * 70)

        print(
            f"Features ({len(usable_features)}): "
            f"{usable_features}"
        )

        train_proxy = train_df[
            usable_features
        ].fillna(0).astype(float).values

        test_proxy = test_df[
            usable_features
        ].fillna(0).astype(float).values

        metrics = train_and_evaluate(
            X_train_text,
            X_test_text,
            y_train,
            y_test,
            train_proxy,
            test_proxy,
        )

        metrics["experiment"] = experiment_name
        metrics["proxy_feature_count"] = len(
            usable_features
        )
        metrics["total_feature_count"] = (
            metrics["tfidf_feature_count"]
            + len(usable_features)
        )

        results.append(metrics)

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )
        print(
            f"Macro-F1:  {metrics['macro_f1']:.4f}"
        )
        print(
            f"Weighted-F1:{metrics['weighted_f1']:.4f}"
        )

    # -----------------------------------------------------
    # 3. Full proxy model
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("Experiment: Full_Model")
    print("-" * 70)

    train_proxy = train_df[
        available_proxy_features
    ].fillna(0).astype(float).values

    test_proxy = test_df[
        available_proxy_features
    ].fillna(0).astype(float).values

    full_model = train_and_evaluate(
        X_train_text,
        X_test_text,
        y_train,
        y_test,
        train_proxy,
        test_proxy,
    )

    full_model["experiment"] = "Full_Model"
    full_model["proxy_feature_count"] = len(
        available_proxy_features
    )
    full_model["total_feature_count"] = (
        full_model["tfidf_feature_count"]
        + len(available_proxy_features)
    )

    results.append(full_model)

    print(
        f"Accuracy:  {full_model['accuracy']:.4f}"
    )
    print(
        f"Macro-F1:  {full_model['macro_f1']:.4f}"
    )
    print(
        f"Weighted-F1:{full_model['weighted_f1']:.4f}"
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    # Put experiment first
    column_order = [
        "experiment",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "proxy_feature_count",
        "tfidf_feature_count",
        "total_feature_count",
    ]

    results_df = results_df[
        [
            col
            for col in column_order
            if col in results_df.columns
        ]
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL ABLATION RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()