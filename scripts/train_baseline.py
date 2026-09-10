import os
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


INPUT_PATH = "data/processed/corpus_processed.csv"
METRICS_PATH = "reports/baseline_metrics.json"
FIGURE_PATH = "reports/figures/confusion_matrix.png"


def create_baseline_model():
    """
    Creates the TF-IDF vectorizer and Logistic Regression classifier.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    return vectorizer, classifier


def main():

    print("=" * 60)
    print("TRAINING BASELINE MODEL")
    print("=" * 60)

    # --------------------------------------------------
    # Load processed dataset
    # --------------------------------------------------

    print("\nLoading processed dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # Validate columns
    # --------------------------------------------------

    required_columns = [
        "clean_text",
        "label",
        "split"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found."
            )

    # --------------------------------------------------
    # Extract train and test sets
    # --------------------------------------------------

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"\nTraining examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Training or test set is empty."
        )

    # --------------------------------------------------
    # Display label distributions
    # --------------------------------------------------

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    # --------------------------------------------------
    # Prepare text and labels
    # --------------------------------------------------

    X_train = train_df["clean_text"].astype(str)
    y_train = train_df["label"]

    X_test = test_df["clean_text"].astype(str)
    y_test = test_df["label"]

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    vectorizer, classifier = create_baseline_model()

    # --------------------------------------------------
    # TF-IDF transformation
    # --------------------------------------------------

    print("\nFitting TF-IDF vectorizer...")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"Training TF-IDF shape: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"Test TF-IDF shape: "
        f"{X_test_tfidf.shape}"
    )

    # --------------------------------------------------
    # Train classifier
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    classifier.fit(
        X_train_tfidf,
        y_train
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = classifier.predict(
        X_test_tfidf
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    labels = [
        "negative",
        "neutral",
        "positive"
    ]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(FIGURE_PATH),
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    ax.set_title(
        "Baseline TF-IDF + Logistic Regression"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"\nConfusion matrix saved to:\n"
        f"{FIGURE_PATH}"
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "tfidf_features": int(
            X_train_tfidf.shape[1]
        )
    }

    os.makedirs(
        os.path.dirname(METRICS_PATH),
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"Baseline metrics saved to:\n"
        f"{METRICS_PATH}"
    )

    print("\nBaseline training complete!")


if __name__ == "__main__":
    main()