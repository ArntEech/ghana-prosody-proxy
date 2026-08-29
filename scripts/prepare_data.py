from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "asante_twi_senti"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(filename, language, dataset_type):
    """
    Load an AsanteTwiSenti dataset and convert it into
    the project's common schema.
    """

    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Loaded {filename}: {len(df)} rows")

    # Rename dataset columns to project-standard names
    df = df.rename(
        columns={
            "tweets": "tweet",
            "labels": "label"
        }
    )

    # Keep required columns
    df = df[["tweet", "label"]].copy()

    # Add metadata
    df["language"] = language
    df["dataset_type"] = dataset_type
    df["dataset"] = "asante_twi_senti"

    return df


# ============================================================
# LOAD TWI DATA
# ============================================================

twi_df = load_dataset(
    filename="labeled_twi_tweets.csv",
    language="twi",
    dataset_type="monolingual"
)


# ============================================================
# LOAD GHANAIAN PIDGIN DATA
# ============================================================

pidgin_df = load_dataset(
    filename="labeled_ghana_pidgin_tweets.csv",
    language="ghanaian_pidgin",
    dataset_type="pidgin"
)


# ============================================================
# LOAD MULTILINGUAL DATA
# ============================================================

multilingual_df = load_dataset(
    filename="labeled_multilingual_tweets.csv",
    language="mixed",
    dataset_type="multilingual"
)


# ============================================================
# COMBINE DATASETS
# ============================================================

combined_df = pd.concat(
    [
        twi_df,
        pidgin_df,
        multilingual_df
    ],
    ignore_index=True
)


# ============================================================
# CLEAN DATA
# ============================================================

# Remove missing tweets or labels
combined_df = combined_df.dropna(
    subset=["tweet", "label"]
)

# Convert tweets to strings
combined_df["tweet"] = (
    combined_df["tweet"]
    .astype(str)
    .str.strip()
)

# Remove empty tweets
combined_df = combined_df[
    combined_df["tweet"].str.len() > 0
]

# Standardise labels
combined_df["label"] = (
    combined_df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# ADD ID
# ============================================================

combined_df.insert(
    0,
    "id",
    range(1, len(combined_df) + 1)
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    PROCESSED_DIR /
    "corpus_processed.csv"
)

combined_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print(f"\nTotal tweets: {len(combined_df)}")

print("\nLanguage distribution:")
print(
    combined_df["language"]
    .value_counts()
)

print("\nDataset type distribution:")
print(
    combined_df["dataset_type"]
    .value_counts()
)

print("\nSentiment distribution:")
print(
    combined_df["label"]
    .value_counts()
)

print(f"\nSaved processed corpus to:")
print(output_path)