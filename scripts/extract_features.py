from pathlib import Path
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from src.proxy_features import extract_proxy_features


INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "corpus_processed.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "corpus_with_proxy_features.csv"
)


def main():

    print("Loading processed corpus...")

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Loaded {len(df)} tweets."
    )

    print(
        "Extracting prosodic and pragmatic proxy features..."
    )

    feature_rows = []

    for index, row in df.iterrows():

        features = extract_proxy_features(
            row["tweet"]
        )

        feature_rows.append(features)

        if (index + 1) % 1000 == 0:
            print(
                f"Processed {index + 1} tweets..."
            )


    feature_df = pd.DataFrame(
        feature_rows
    )

    final_df = pd.concat(
        [
            df.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1
    )


    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )


    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"\nProcessed tweets: {len(final_df)}"
    )

    print(
        "\nTweets containing elongation:"
    )

    print(
        final_df["has_elongation"]
        .sum()
    )

    print(
        "\nTweets containing emojis:"
    )

    print(
        final_df["has_emoji"]
        .sum()
    )

    print(
        "\nTweets containing pragmatic particles:"
    )

    print(
        final_df["has_particle"]
        .sum()
    )

    print(
        "\nTweets containing interjections:"
    )

    print(
        final_df["has_interjection"]
        .sum()
    )

    print(
        f"\nSaved output to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()