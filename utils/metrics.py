import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(model, dataset, label_lookup):
    """
    Comprehensive evaluation for classification model.
    Includes:
    - Accuracy
    - Precision (macro & weighted)
    - Recall (macro & weighted)
    - F1-score (macro & weighted)
    - Classification report
    - Confusion matrix
    """

    y_true = []
    y_pred = []

    for x_batch, y_batch in dataset:
        predictions = model.predict(x_batch, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(y_batch.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n================ MODEL EVALUATION ================\n")

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy              : {accuracy:.4f}")
    print(f"Precision (Macro)     : {precision_macro:.4f}")
    print(f"Recall (Macro)        : {recall_macro:.4f}")
    print(f"F1-Score (Macro)      : {f1_macro:.4f}")
    print(f"Precision (Weighted)  : {precision_weighted:.4f}")
    print(f"Recall (Weighted)     : {recall_weighted:.4f}")
    print(f"F1-Score (Weighted)   : {f1_weighted:.4f}\n")

    unique_labels = np.unique(y_true)
    target_names = [
        label_lookup.get_vocabulary()[i]
        for i in unique_labels
    ]

    print("Classification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0
        )
    )

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    print("\n==================================================\n")