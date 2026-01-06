"""
Text LSTM trainer compatible with:
pandas==2.3.3
numpy==1.26.4
tensorflow==2.11
(+ Flask/opencv/joblib present but not used here)

Changes vs original:
- Removes scikit-learn imports (train_test_split + metrics) because sklearn isn't in requirements.txt
- Replaces TimeTracker with time.perf_counter
- Implements a simple stratified train/test split in numpy
- Computes basic binary metrics without sklearn
- Saves model as .h5
- Saves tokenizer as pickle
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/data/AI_Human_cleaned.csv")

MAX_NB_WORDS = 20_000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100

TEST_SIZE = 0.2
RANDOM_SEED = 0

BATCH_SIZE = 64
EPOCHS = 1
VAL_SPLIT = 0.1

MODEL_OUT = Path("lstm_model.h5")
TOKENIZER_OUT = Path("tokenizer.pickle")


# -----------------------------
# Utils (no sklearn)
# -----------------------------
def normalize_binary_labels(y: pd.Series) -> np.ndarray:
    """
    Accepts common binary label formats:
      - 0/1 ints
      - bools
      - strings like "ai"/"human" or any 2 unique strings

    Returns np.int32 array with values in {0,1}.
    """
    # numeric?
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().all():
        uniq = sorted(set(y_num.astype(int).tolist()))
        if uniq == [0, 1]:
            return y_num.astype(np.int32).to_numpy()

    y_str = y.astype(str).str.strip().str.lower()
    uniq = sorted(y_str.unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"Expected 2 unique labels, got {len(uniq)}: {uniq}")

    # heuristic mapping
    if "human" in uniq and "ai" in uniq:
        mapping = {"human": 0, "ai": 1}
    else:
        mapping = {uniq[0]: 0, uniq[1]: 1}

    return y_str.map(mapping).astype(np.int32).to_numpy()


def stratified_split_indices(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns train_idx, test_idx with stratification for binary labels.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Stratified split expects 2 classes; got {classes}")

    train_parts = []
    test_parts = []

    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    test_idx = np.concatenate(test_parts)
    train_idx = np.concatenate(train_parts)
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t0 = time.perf_counter()

    print(">>> Data loading...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    dataset = pd.read_csv(DATA_PATH)
    if dataset.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns (text,label). Got: {dataset.columns.tolist()}")

    text_col = dataset.columns[0]
    label_col = dataset.columns[1]

    print(">>> Cleaning...")
    dataset = dataset.dropna(subset=[text_col, label_col])
    dataset = dataset.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    print(f"DATA NUMBER: {len(dataset)}")

    x_text = dataset[text_col].astype(str).to_numpy()
    y = normalize_binary_labels(dataset[label_col])

    print(">>> Train/test split...")
    train_idx, test_idx = stratified_split_indices(y, test_size=TEST_SIZE, seed=RANDOM_SEED)
    x_train_text, x_test_text = x_text[train_idx], x_text[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Leakage control (string-based exact duplicates)
    print("Common text between test and train:", len(set(x_train_text.tolist()) & set(x_test_text.tolist())))

    print(">>> Tokenization...")
    tokenizer = Tokenizer(
        num_words=MAX_NB_WORDS,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
        lower=True,
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(x_train_text.tolist())

    x_train_seq = tokenizer.texts_to_sequences(x_train_text.tolist())
    x_test_seq = tokenizer.texts_to_sequences(x_test_text.tolist())

    x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
    x_test_pad = pad_sequences(x_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

    print(f"Train shape: {x_train_pad.shape}")
    print(f"Test  shape: {x_test_pad.shape}")

    print(">>> Creating model...")
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    print(">>> Training...")
    model.fit(
        x_train_pad,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        verbose=1,
    )

    print(">>> Predict/Evaluate...")
    y_pred_probs = model.predict(x_test_pad, verbose=0).reshape(-1)
    y_pred = (y_pred_probs >= 0.5).astype(np.int32)

    m = binary_metrics(y_test, y_pred)
    print("-" * 60)
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1       : {m['f1']:.4f}")
    print("Confusion matrix (tn fp / fn tp):")
    print(f"{m['tn']:6d} {m['fp']:6d}")
    print(f"{m['fn']:6d} {m['tp']:6d}")
    print("-" * 60)

    print(">>> Saving model (.h5)...")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT.as_posix())
    print(f"Saved model: {MODEL_OUT.resolve()}")

    print(">>> Saving tokenizer (pickle)...")
    with open(TOKENIZER_OUT, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved tokenizer: {TOKENIZER_OUT.resolve()}")

    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    # Optional: reduce TF noise
    tf.get_logger().setLevel("ERROR")
    main()
