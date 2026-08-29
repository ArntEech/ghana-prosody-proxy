from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.proxy_features import extract_proxy_features
from src.language_features import detect_language_features


INPUT_FILE = ROOT / "data/processed/corpus_processed.csv"
OUTPUT_FILE = ROOT / "data/annotated/prosody_proxy_annotations.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    if "text" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'text' column."
        )

    proxy_rows = []

    for text in df["text"]:
        features = {}

        features.update(
            extract_proxy_features(text)
        )

        features.update(
            detect_language_features(text)
        )

        proxy_rows.append(features)

    feature_df = pd.DataFrame(proxy_rows)

    output = pd.concat(
        [df.reset_index(drop=True), feature_df],
        axis=1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature extraction complete: "
        f"{len(output)} records"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()