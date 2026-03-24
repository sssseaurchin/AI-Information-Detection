import math

import numpy as np
import tensorflow as tf


def brier_score_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the binary Brier score from positive-class probabilities."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean(np.square(y_prob - y_true)))


def reliability_diagram_binary(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> dict[str, list[dict[str, float]] | float]:
    """Compute reliability bin statistics and expected calibration error."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    total = max(1, len(y_prob))
    bin_rows: list[dict[str, float]] = []
    ece = 0.0

    for idx in range(num_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == num_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(np.sum(mask))
        if count == 0:
            bin_rows.append(
                {
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_confidence = float(np.mean(y_prob[mask]))
        avg_accuracy = float(np.mean(y_true[mask]))
        gap = abs(avg_accuracy - avg_confidence)
        ece += (count / total) * gap
        bin_rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "gap": float(gap),
            }
        )

    return {"ece": float(ece), "bins": bin_rows}


def _probabilities_to_binary_logits(positive_probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(positive_probs, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_temperature(
    labels: np.ndarray,
    logits: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    learning_rate: float = 0.05,
    steps: int = 200,
) -> dict[str, float | str]:
    """Fit a single temperature scalar on validation or calibration data."""
    y_true = tf.convert_to_tensor(labels.astype(np.int32))

    if logits is not None:
        raw_logits = tf.convert_to_tensor(logits.astype(np.float32))
        method = "logits"
    elif probabilities is not None:
        positive_logits = _probabilities_to_binary_logits(np.asarray(probabilities, dtype=float))
        raw_logits = tf.convert_to_tensor(np.stack([-positive_logits, positive_logits], axis=1).astype(np.float32))
        method = "probability_space_warning"
    else:
        raise ValueError("fit_temperature requires either logits or probabilities.")

    log_temperature = tf.Variable(0.0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            temperature = tf.exp(log_temperature)
            scaled_logits = raw_logits / temperature
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_true, scaled_logits, from_logits=True)
            )
        grads = tape.gradient(loss, [log_temperature])
        optimizer.apply_gradients(zip(grads, [log_temperature]))

    final_temperature = float(tf.exp(log_temperature).numpy())
    final_loss = float(
        tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_true, raw_logits / tf.exp(log_temperature), from_logits=True
            )
        ).numpy()
    )
    return {
        "temperature": final_temperature,
        "nll": final_loss,
        "method": method,
    }


def apply_temperature_scaling(
    logits: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    temperature: float = 1.0,
) -> dict[str, np.ndarray | str]:
    """Apply temperature scaling to logits if available, otherwise fall back to probability-space scaling."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if logits is not None:
        scaled_logits = np.asarray(logits, dtype=float) / temperature
        scaled_probs = tf.nn.softmax(tf.convert_to_tensor(scaled_logits, dtype=tf.float32), axis=1).numpy()
        return {"probabilities": scaled_probs, "method": "logits"}

    if probabilities is None:
        raise ValueError("apply_temperature_scaling requires logits or probabilities.")

    positive_logits = _probabilities_to_binary_logits(np.asarray(probabilities, dtype=float))
    scaled_positive = 1.0 / (1.0 + np.exp(-(positive_logits / temperature)))
    scaled_probs = np.stack([1.0 - scaled_positive, scaled_positive], axis=1)
    return {"probabilities": scaled_probs, "method": "probability_space_warning"}
