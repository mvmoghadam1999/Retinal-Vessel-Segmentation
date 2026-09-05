import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    jaccard_score, precision_score, recall_score
)


def binary_segmentation_metrics(targets, predictions, probabilities):
    targets = targets.astype(np.uint8).reshape(-1)
    predictions = predictions.astype(np.uint8).reshape(-1)
    probabilities = probabilities.astype(np.float32).reshape(-1)

    intersection = np.logical_and(targets, predictions).sum()
    dice = 2.0 * intersection / (targets.sum() + predictions.sum() + 1e-8)

    return {
        "Accuracy": accuracy_score(targets, predictions),
        "Precision": precision_score(targets, predictions, zero_division=0),
        "Recall": recall_score(targets, predictions, zero_division=0),
        "F1": f1_score(targets, predictions, zero_division=0),
        "Dice": dice,
        "IoU": jaccard_score(targets, predictions, average="binary", zero_division=0),
        "mAP": average_precision_score(targets, probabilities),
    }
