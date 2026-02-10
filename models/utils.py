from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Compute all required evaluation metrics.
    Returns a dictionary.
    """
    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    if y_prob is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["AUC"] = None

    return metrics
