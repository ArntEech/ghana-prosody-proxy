from pathlib import Path
import sys
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline_model import build_baseline_model


INPUT_FILE = ROOT / "data/annotated/prosody_proxy_annotations.csv"
OUTPUT_FILE = ROOT / "reports/baseline_metrics.json"


def main():

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(
        subset=["text", "label"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    vectorizer, model = build_baseline_model()

    X_train_vec = vectorizer.fit_transform(
        X_train
    )

    X_test_vec = vectorizer.transform(
        X_test
    )

    model.fit(
        X_train_vec,
        y_train
    )

    predictions = model.predict(
        X_test_vec
    )

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro"
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test,
                predictions,
                average="weighted"
            )
        ),
        "n_train": len(X_train),
        "n_test": len(X_test)
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

    print(json.dumps(
        metrics,
        indent=4
    ))


if __name__ == "__main__":
    main()