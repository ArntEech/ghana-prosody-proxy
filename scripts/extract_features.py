from pathlib import Path
import sys

import pandas as pd


# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import preprocess_text
from src.proxy_features import extract_proxy_features


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "corpus_processed.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "corpus_with_features.csv"
)


def main():

    print("Loading processed corpus...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} records.")

    print("Extracting prosodic and pragmatic proxy features...")

    feature_rows = []

    for _, row in df.iterrows():

        text = preprocess_text(row["text"])

        features = extract_proxy_features(text)

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    final_df = pd.concat(
        [
            df.reset_index(drop=True),
            features_df.reset_index(drop=True)
        ],
        axis=1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature extraction complete."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nExtracted features:")

    print(features_df.columns.tolist())


if __name__ == "__main__":
    main()