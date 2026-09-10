from pathlib import Path
import sys
import json

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.enhanced_model import (
    build_proxy_model,
    combine_features,
    PROXY_COLUMNS
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = ROOT / "data/processed/corpus_with_features.csv"
OUTPUT_FILE = ROOT / "reports/proxy_metrics.json"


def main():

    print("=" * 60)
    print("TRAINING PROXY-ENHANCED MODEL")
    print("=" * 60)

    # --------------------------------------------------
    # Load processed corpus
    # --------------------------------------------------

    print("\nLoading corpus with proxy features...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------

    required_columns = [
        "clean_text",
        "label",
        "split"
    ] + PROXY_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------
    # Remove missing values
    # --------------------------------------------------

    df = df.dropna(
        subset=["clean_text", "label"]
    ).copy()

    # --------------------------------------------------
    # Use the SAME splits as the baseline
    # --------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        f"\nTraining examples: "
        f"{len(train_df)}"
    )

    print(
        f"Test examples: "
        f"{len(test_df)}"
    )

    # --------------------------------------------------
    # Display label distributions
    # --------------------------------------------------

    print("\nTraining label distribution:")

    print(
        train_df["label"].value_counts()
    )

    print("\nTest label distribution:")

    print(
        test_df["label"].value_counts()
    )

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    print(
        "\nBuilding proxy-enhanced model..."
    )

    vectorizer, model = build_proxy_model()

    # --------------------------------------------------
    # TF-IDF features
    # --------------------------------------------------

    print(
        "\nExtracting TF-IDF features..."
    )

    X_train_text = vectorizer.fit_transform(
        train_df["clean_text"].astype(str)
    )

    X_test_text = vectorizer.transform(
        test_df["clean_text"].astype(str)
    )

    print(
        f"Training TF-IDF shape: "
        f"{X_train_text.shape}"
    )

    print(
        f"Test TF-IDF shape: "
        f"{X_test_text.shape}"
    )

    # --------------------------------------------------
    # Proxy features
    # --------------------------------------------------

    print(
        "\nPreparing proxy features..."
    )

    X_train_proxy = csr_matrix(
        train_df[PROXY_COLUMNS]
        .fillna(0)
        .astype(float)
        .values
    )

    X_test_proxy = csr_matrix(
        test_df[PROXY_COLUMNS]
        .fillna(0)
        .astype(float)
        .values
    )

    print(
        f"Proxy feature count: "
        f"{len(PROXY_COLUMNS)}"
    )

    # --------------------------------------------------
    # Combine TF-IDF + proxy features
    # --------------------------------------------------

    print(
        "\nCombining TF-IDF and proxy features..."
    )

    X_train = combine_features(
        X_train_text,
        X_train_proxy
    )

    X_test = combine_features(
        X_test_text,
        X_test_proxy
    )

    print(
        f"Combined training shape: "
        f"{X_train.shape}"
    )

    print(
        f"Combined test shape: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model.fit(
        X_train,
        train_df["label"]
    )

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(
        test_df["label"],
        predictions
    )

    macro_f1 = f1_score(
        test_df["label"],
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        test_df["label"],
        predictions,
        average="weighted"
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PROXY-ENHANCED MODEL RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro-F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics = {

        "model":
            "TF-IDF + Textual Prosodic Proxy Features + Logistic Regression",

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "n_train":
            int(len(train_df)),

        "n_test":
            int(len(test_df)),

        "tfidf_feature_count":
            int(X_train_text.shape[1]),

        "proxy_feature_count":
            int(len(PROXY_COLUMNS)),

        "total_feature_count":
            int(X_train.shape[1])
    }

    OUTPUT_FILE.parent.mkdir(
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nProxy-enhanced training complete!"
    )


if __name__ == "__main__":
    main()