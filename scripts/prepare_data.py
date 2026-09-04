import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed/corpus_processed.csv"
OUTPUT_PATH = "data/processed/corpus_processed.csv"
EXCLUDED_PATH = "data/processed/excluded_labels.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VALID_LABELS = [
    "positive",
    "negative",
    "neutral"
]


def clean_text(text):
    """
    Performs light cleaning while preserving features
    important for prosodic and pragmatic analysis.

    Preserved:
    - emojis
    - repeated characters
    - capitalization
    - punctuation
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Replace URLs
    text = re.sub(
        r"http\S+|www\S+",
        "<URL>",
        text
    )

    # Replace user mentions
    text = re.sub(
        r"@\w+",
        "<USER>",
        text
    )

    # Normalize excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def main():

    print("=" * 60)
    print("PREPARING ASANTETWISENTI DATASET")
    print("=" * 60)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original dataset shape: {df.shape}")

    print("\nColumns found:")
    print(df.columns.tolist())

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------

    required_columns = ["text", "label"]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' not found."
            )

    # --------------------------------------------------
    # Inspect original labels
    # --------------------------------------------------

    print("\nOriginal label distribution:")

    print(
        df["label"].value_counts()
    )

    # --------------------------------------------------
    # Remove missing text/labels
    # --------------------------------------------------

    df = df.dropna(
        subset=["text", "label"]
    ).copy()

    print(
        f"\nShape after removing missing values: "
        f"{df.shape}"
    )

    # --------------------------------------------------
    # Create clean text
    # --------------------------------------------------

    print("\nCreating clean_text column...")

    df["clean_text"] = df["text"].apply(
        clean_text
    )

    # Remove empty text

    df = df[
        df["clean_text"].str.strip() != ""
    ].copy()

    # --------------------------------------------------
    # Keep only valid sentiment labels
    # --------------------------------------------------

    print("\nFiltering sentiment labels...")

    valid_mask = df["label"].isin(
        VALID_LABELS
    )

    excluded_df = df[
        ~valid_mask
    ].copy()

    df = df[
        valid_mask
    ].copy()

    print(
        f"Retained sentiment examples: {len(df)}"
    )

    print(
        f"Excluded non-standard examples: "
        f"{len(excluded_df)}"
    )

    # --------------------------------------------------
    # Save excluded records for transparency
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(EXCLUDED_PATH),
        exist_ok=True
    )

    excluded_df.to_csv(
        EXCLUDED_PATH,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nExcluded records saved to:\n"
        f"{EXCLUDED_PATH}"
    )

    # --------------------------------------------------
    # Display clean label distribution
    # --------------------------------------------------

    print("\nClean sentiment distribution:")

    print(
        df["label"].value_counts()
    )

    # --------------------------------------------------
    # Create train/dev/test split
    # --------------------------------------------------

    print(
        "\nCreating stratified "
        "train/dev/test splits..."
    )

    # 70% training
    # 30% temporary

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    # Split remaining 30% into:
    # 15% development
    # 15% test

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    # --------------------------------------------------
    # Add split labels
    # --------------------------------------------------

    train_df = train_df.copy()
    dev_df = dev_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    dev_df["split"] = "dev"
    test_df["split"] = "test"

    # --------------------------------------------------
    # Combine datasets
    # --------------------------------------------------

    final_df = pd.concat(
        [
            train_df,
            dev_df,
            test_df
        ],
        ignore_index=True
    )

    # Shuffle

    final_df = final_df.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"\nFinal dataset shape: "
        f"{final_df.shape}"
    )

    print("\nSplit distribution:")

    print(
        final_df["split"].value_counts()
    )

    print("\nLabel distribution:")

    print(
        final_df["label"].value_counts()
    )

    print("\nLabel distribution by split:")

    print(
        pd.crosstab(
            final_df["split"],
            final_df["label"]
        )
    )

    print("\nFinal columns:")

    print(
        final_df.columns.tolist()
    )

    print(
        f"\nProcessed dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\nPreparation successful!")


if __name__ == "__main__":
    main()