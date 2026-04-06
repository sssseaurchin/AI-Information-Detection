"""
Text LSTM trainer compatible with:
pandas==2.3.3
numpy==1.26.4
tensorflow==2.11

Refactor goals:
- Move script flow into small functions
- Centralize config
- Keep "no sklearn" constraint
- Keep same behavior (cleaning, stratified split, tokenize, train, eval, save)
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, SpatialDropout1D,
    GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D,
    Bidirectional, GRU, Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#CONTROL PANEL
 
data_path: Path = Path("data_center/data/AI Generated Essays Dataset.csv")

max_nb_words: int = 20_000          #how many unique words does model will study
max_sequence_length: int = 200      #put something to blank spaces and limit text to (200) sentence
embedding_dim: int = 100            # The number of math variables used for a word.

test_size: float = 0.2      #what percentege is test size
random_seed: int = 0        # Locks the randomness. Ensures data splits the exact same way every time we run

batch_size: int = 64        #how many example will studied before model updates its brain
epochs: int = 1             # How many times the model will read through the ENTIRE dataset.
val_split: float = 0.1

model_out: Path = Path("lstm_model.h5")             #path for saving model's brain
tokenizer_out: Path = Path("tokenizer.pickle")      #path for saving dictionary(words and their numbers)

# training / tokenization knobs
oov_token: str = "<OOV>"        # "Out Of Vocabulary". The placeholder used if the tokenizer sees a brand new word it didn't learn.
tf_filters: str = r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'        # Punctuation marks to delete so "Hello!" and "Hello" are treated as the same word.
threshold: float = 0.5      #for easy choose 1 and 0. like 0.51 and 0.49.

 
def normalize_binary_labels(y: pd.Series) -> np.ndarray:
    """
    Accepts common binary label formats:
      - 0/1 ints
      - bools
      - strings like "ai"/"human" or any 2 unique strings

    Returns np.int32 array with values in {0,1}.
    """
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().all():
        uniq = sorted(set(y_num.astype(int).tolist()))
        if uniq == [0, 1]:
            return y_num.astype(np.int32).to_numpy()

    y_str = y.astype(str).str.strip().str.lower()
    uniq = sorted(y_str.unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"Expected 2 unique labels, got {len(uniq)}: {uniq}")

    if "human" in uniq and "ai" in uniq:
        mapping = {"human": 0, "ai": 1}
    else:
        mapping = {uniq[0]: 0, uniq[1]: 1}

    return y_str.map(mapping).astype(np.int32).to_numpy()


def stratified_split_indices(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns train_idx, test_idx with stratification for binary labels."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Stratified split expects 2 classes; got {classes}")

    train_parts, test_parts = [], []

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





def stratified_kfold_indices(y: np.ndarray, k: int, seed: int):
    """
    Returns list of (train_idx, test_idx) tuples for stratified K-fold.
    Works for binary labels (0/1).
    """
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes, got {classes}")

    # Her sınıf için indexleri ayır
    per_class_folds = {}

    for c in classes:
        idx = np.where(y == c)[0]  # o class'a ait indexler
        rng.shuffle(idx)           # karıştır
        folds = np.array_split(idx, k)  # k parçaya böl
        per_class_folds[c] = folds

    splits = []

    for fold_i in range(k):
        test_parts = []
        train_parts = []

        for c in classes:
            folds = per_class_folds[c]

            # fold_i test olacak
            test_parts.append(folds[fold_i])

            # diğerleri train
            train_parts.append(
                np.concatenate([folds[j] for j in range(k) if j != fold_i])
            )

        test_idx = np.concatenate(test_parts)
        train_idx = np.concatenate(train_parts)

        rng.shuffle(test_idx)
        rng.shuffle(train_idx)

        splits.append((train_idx, test_idx))

    return splits











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
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def exact_text_overlap_count(train_text: np.ndarray, test_text: np.ndarray) -> int:
    # Leakage check: exact string matches across splits.
    return len(set(train_text.tolist()) & set(test_text.tolist()))


# -----------------------------
# Data
# -----------------------------
def load_dataset() -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")
    return pd.read_csv(data_path)


def infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns (text,label). Got: {df.columns.tolist()}")
    return df.columns[0], df.columns[1]


def clean_dataset(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    return df


def build_xy(df: pd.DataFrame, text_col: str, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    x_text = df[text_col].astype(str).to_numpy()
    y = normalize_binary_labels(df[label_col])
    return x_text, y


def train_test_split_stratified(
    x_text: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, test_idx = stratified_split_indices(y, test_size=test_size, seed=seed)
    return x_text[train_idx], x_text[test_idx], y[train_idx], y[test_idx]

 
def build_tokenizer() -> Tokenizer:
    return Tokenizer(
        num_words=max_nb_words,
        filters=tf_filters,
        lower=True,
        oov_token=oov_token,
    )


def fit_and_transform_text(
    tokenizer: Tokenizer,
    x_train_text: np.ndarray,
    x_test_text: np.ndarray,
    max_sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer.fit_on_texts(x_train_text.tolist())

    x_train_seq = tokenizer.texts_to_sequences(x_train_text.tolist())
    x_test_seq = tokenizer.texts_to_sequences(x_test_text.tolist())

    x_train_pad = pad_sequences(x_train_seq, maxlen=max_sequence_length)
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_sequence_length)

    return x_train_pad, x_test_pad


# Model Build
def build_model(model_name: str = "lstm") -> tf.keras.Model:
    if model_name == "avgpool":
        model = Sequential([
            Embedding(max_nb_words, embedding_dim, input_length=max_sequence_length),
            GlobalAveragePooling1D(),
            Dense(1, activation="sigmoid"),
        ])


    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_model(
    model: tf.keras.Model,
    x_train_pad: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    epochs: int,
    val_split: float,
) -> None:
    model.fit(
        x_train_pad,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=1,
    )


def predict_labels(model: tf.keras.Model, x_test_pad: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    y_pred_probs = model.predict(x_test_pad, verbose=0).reshape(-1)
    y_pred = (y_pred_probs >= threshold).astype(np.int32)
    return y_pred_probs, y_pred




def run_one_fold(
    model_name: str,
    x_text: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, float]:
    # (Öneri) TF bellek/graph temizliği
    tf.keras.backend.clear_session()

    x_train_text = x_text[train_idx]
    x_test_text = x_text[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Tokenizer her fold'da sadece train'e fit edilmeli (leakage yok)
    tokenizer = build_tokenizer()
    x_train_pad, x_test_pad = fit_and_transform_text(
        tokenizer=tokenizer,
        x_train_text=x_train_text,
        x_test_text=x_test_text,
        max_sequence_length=max_sequence_length,
    )

    # Model seçimi
    model = build_model(model_name)

    # Train
    train_model(
        model=model,
        x_train_pad=x_train_pad,
        y_train=y_train,
        batch_size=batch_size,
        epochs=epochs,
        val_split=val_split,
    )

    # Eval
    _, y_pred = predict_labels(model, x_test_pad, threshold=threshold)
    m = binary_metrics(y_test, y_pred)
    return m



def run_cv(model_name: str, k: int = 5) -> None:
    t0 = time.perf_counter()
    print(f"\n########## CV START: {model_name} | k={k} ##########")

    df = load_dataset()
    text_col, label_col = infer_columns(df)

    df = clean_dataset(df, text_col, label_col)
    print(f"DATA NUMBER: {len(df)}")

    x_text, y = build_xy(df, text_col, label_col)

    # K-fold split üret
    splits = stratified_kfold_indices(y, k=k, seed=random_seed)

    metrics_list = []
    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n=== {model_name} | Fold {fold_i}/{k} ===")
        print("Leakage (exact text overlap):", exact_text_overlap_count(x_text[train_idx], x_text[test_idx]))

        m = run_one_fold(model_name, x_text, y, train_idx, test_idx)
        print_metrics(m)
        metrics_list.append(m)

    # Ortalama + std
    for key in ["accuracy", "precision", "recall", "f1"]:
        vals = np.array([m[key] for m in metrics_list], dtype=float)
        print(f"{model_name} {key}: mean={vals.mean():.4f} std={vals.std():.4f}")

    elapsed = time.perf_counter() - t0
    print(f"Total CV time: {elapsed:.2f}s")
    print(f"########## CV END: {model_name} ##########\n")


# -----------------------------
# Save/Load
# -----------------------------
def save_model(model: tf.keras.Model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path.as_posix())


def save_tokenizer(tokenizer: Tokenizer, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

 
def print_metrics(m: Dict[str, float]) -> None:
    print("-" * 60)
    print(f"Accuracy : {m['accuracy']}")
    print(f"Precision: {m['precision']}")
    print(f"Recall   : {m['recall']}")
    print(f"F1       : {m['f1']}")
    print("Confusion matrix (tn fp / fn tp):")
    print(f"{int(m['tn']):6d} {int(m['fp']):6d}")
    print(f"{int(m['fn']):6d} {int(m['tp']):6d}")
    print("-" * 60)
 
def run(model_name: str = "lstm") -> None:
    t0 = time.perf_counter()

    print(">>> Data loading...")
    df = load_dataset()
    text_col, label_col = infer_columns(df)

    print(">>> Cleaning...")
    df = clean_dataset(df, text_col, label_col)
    print(f"DATA NUMBER: {len(df)}")

    x_text, y = build_xy(df, text_col, label_col)

    print(">>> Train/test split...")
    x_train_text, x_test_text, y_train, y_test = train_test_split_stratified(
        x_text=x_text,
        y=y,
        test_size=test_size,
        seed=random_seed,
    )
    print("Common text between test and train:", exact_text_overlap_count(x_train_text, x_test_text))

    print(">>> Tokenization...")
    tokenizer = build_tokenizer()
    x_train_pad, x_test_pad = fit_and_transform_text(
        tokenizer=tokenizer,
        x_train_text=x_train_text,
        x_test_text=x_test_text,
        max_sequence_length=max_sequence_length,
    )
    print(f"Train shape: {x_train_pad.shape}")
    print(f"Test  shape: {x_test_pad.shape}")

    print(f">>> Creating model: {model_name} ...")
    model = build_model(model_name)   # ✅ artık model_name var
    model.summary()

    print(">>> Training...")
    train_model(
        model=model,
        x_train_pad=x_train_pad,
        y_train=y_train,
        batch_size=batch_size,
        epochs=epochs,
        val_split=val_split,
    )

    print(">>> Predict/Evaluate...")
    _, y_pred = predict_labels(model, x_test_pad, threshold=threshold)
    m = binary_metrics(y_test, y_pred)
    print_metrics(m)

    # model/tokenizer kaydetmeyi istersen model adına göre ayırmak mantıklı:
    model_path = model_out.with_name(f"{model_name}_{model_out.name}")
    tok_path = tokenizer_out.with_name(f"{model_name}_{tokenizer_out.name}")

    print(">>> Saving model (.h5)...")
    save_model(model, model_path)
    print(f"Saved model: {model_path.resolve()}")

    print(">>> Saving tokenizer (pickle)...")
    save_tokenizer(tokenizer, tok_path)
    print(f"Saved tokenizer: {tok_path.resolve()}")

    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.2f}s")


    

def main() -> None:
    tf.get_logger().setLevel("ERROR")
    
    run_cv("avgpool")

if __name__ == "__main__":
    main()
