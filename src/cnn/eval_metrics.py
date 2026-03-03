import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return the binary confusion matrix counts."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | dict[str, int]]:
    """Compute thresholded binary classification metrics at a fixed operating point."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix_binary(y_true, y_pred)

    precision = _safe_divide(cm["tp"], cm["tp"] + cm["fp"])
    recall = _safe_divide(cm["tp"], cm["tp"] + cm["fn"])
    specificity = _safe_divide(cm["tn"], cm["tn"] + cm["fp"])
    accuracy = _safe_divide(cm["tp"] + cm["tn"], len(y_true))
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    """Compute ROC curve points and thresholds for binary classification."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    thresholds = np.r_[np.inf, np.unique(y_score_sorted)]
    positives = max(1, int(np.sum(y_true == 1)))
    negatives = max(1, int(np.sum(y_true == 0)))

    tpr_values = [0.0]
    fpr_values = [0.0]
    threshold_values = [float("inf")]

    for threshold in thresholds[1:]:
        predicted_positive = y_score >= threshold
        tp = np.sum((predicted_positive == 1) & (y_true == 1))
        fp = np.sum((predicted_positive == 1) & (y_true == 0))
        tpr_values.append(float(tp / positives))
        fpr_values.append(float(fp / negatives))
        threshold_values.append(float(threshold))

    if tpr_values[-1] != 1.0 or fpr_values[-1] != 1.0:
        tpr_values.append(1.0)
        fpr_values.append(1.0)
        threshold_values.append(float(np.min(y_score_sorted) - 1e-12))

    return {
        "fpr": fpr_values,
        "tpr": tpr_values,
        "thresholds": threshold_values,
    }


def pr_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    """Compute precision-recall curve points and thresholds for binary classification."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.unique(np.sort(y_score))[::-1]
    precision_values = [1.0]
    recall_values = [0.0]
    threshold_values = [float("inf")]

    positives = max(1, int(np.sum(y_true == 1)))
    for threshold in thresholds:
        predicted_positive = y_score >= threshold
        tp = np.sum((predicted_positive == 1) & (y_true == 1))
        fp = np.sum((predicted_positive == 1) & (y_true == 0))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, positives)
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        threshold_values.append(float(threshold))

    if recall_values[-1] != 1.0:
        precision_values.append(float(np.mean(y_true)))
        recall_values.append(1.0)
        threshold_values.append(float(np.min(y_score) - 1e-12))

    return {
        "precision": precision_values,
        "recall": recall_values,
        "thresholds": threshold_values,
    }


def auc_trapezoid(x_values: list[float], y_values: list[float]) -> float:
    """Compute trapezoidal area under a curve."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC from the ROC curve."""
    curve = roc_curve_binary(y_true, y_score)
    return auc_trapezoid(curve["fpr"], curve["tpr"])


def pr_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC from the precision-recall curve."""
    curve = pr_curve_binary(y_true, y_score)
    return auc_trapezoid(curve["recall"], curve["precision"])


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, policy: str, fixed_threshold: float | None = None) -> dict[str, float]:
    """Select an operating threshold using a validation-only policy."""
    normalized_policy = policy.strip().lower()
    if normalized_policy == "fixed":
        if fixed_threshold is None:
            raise ValueError("fixed threshold policy requires --fixed-threshold.")
        return {"policy": normalized_policy, "threshold": float(fixed_threshold), "objective": float("nan")}

    candidate_thresholds = np.unique(np.asarray(y_score, dtype=float))
    if candidate_thresholds.size == 0:
        raise ValueError("Cannot select a threshold from an empty score array.")

    best_threshold = float(candidate_thresholds[0])
    best_objective = -np.inf

    for threshold in candidate_thresholds:
        metrics = threshold_metrics(y_true, y_score, float(threshold))
        if normalized_policy == "youden":
            tpr = metrics["recall"]
            cm = metrics["confusion_matrix"]
            fpr = _safe_divide(cm["fp"], cm["fp"] + cm["tn"])
            objective = float(tpr - fpr)
        elif normalized_policy == "f1":
            objective = float(metrics["f1"])
        else:
            raise ValueError(f"Unsupported threshold policy: {policy}")

        if objective > best_objective:
            best_objective = objective
            best_threshold = float(threshold)

    return {"policy": normalized_policy, "threshold": best_threshold, "objective": float(best_objective)}


def threshold_free_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Return the threshold-free metrics used in the research evaluation protocol."""
    return {
        "roc_auc": roc_auc_binary(y_true, y_score),
        "pr_auc": pr_auc_binary(y_true, y_score),
    }
