from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_true, y_pred):
    """
    Evaluates model predictions.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "macro_f1": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "weighted_f1": f1_weighted
    }

    return metrics


def get_classification_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        zero_division=0
    )


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)