from pathlib import Path
import sys
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.enhanced_model import (
    build_proxy_model,
    combine_features,
    PROXY_COLUMNS
)


INPUT_FILE = ROOT / "data/annotated/prosody_proxy_annotations.csv"
OUTPUT_FILE = ROOT / "reports/proxy_metrics.json"


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

    vectorizer, model = build_proxy_model()

    X_train_text = vectorizer.fit_transform(
        train_df["text"]
    )

    X_test_text = vectorizer.transform(
        test_df["text"]
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

    X_train = combine_features(
        X_train_text,
        X_train_proxy
    )

    X_test = combine_features(
        X_test_text,
        X_test_proxy
    )

    model.fit(
        X_train,
        train_df["label"]
    )

    predictions = model.predict(
        X_test
    )

    metrics = {
        "model": "TF-IDF + Textual Prosodic Proxy Features + Logistic Regression",
        "accuracy": float(
            accuracy_score(
                test_df["label"],
                predictions
            )
        ),
        "macro_f1": float(
            f1_score(
                test_df["label"],
                predictions,
                average="macro"
            )
        ),
        "weighted_f1": float(
            f1_score(
                test_df["label"],
                predictions,
                average="weighted"
            )
        ),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "proxy_feature_count": len(PROXY_COLUMNS)
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
        json.dumps(
            metrics,
            indent=4
        )
    )


if __name__ == "__main__":
    main()