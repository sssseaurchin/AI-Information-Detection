"""
bilstm2.py — BiLSTM Test & Experiment Pipeline (v2 — Full)
============================================================
5 makaleden çıkan TÜM pratikleri birleştiren test dosyası.

Makalelerden alınan özellikler:
[Wu et al. 2025 Survey]     → Evaluation metrikleri, class weights, multi-domain dataset bilgisi
[Pu et al. 2023 IEEE S&P]  → GloVe embedding, semantic feature önemi, EarlyStopping
[SA-BiLSTM 2024 IEEE]      → Self-Attention layer, GloVe, stop word removal testi
[Dang et al. 2021 ICISSP]  → Progressive comparison, 5-run mean±std, embedding etkisi, CNN+BiLSTM
[Chang 2025 WMSCI]         → ML baseline, feature importance, %10 veriyle %92.8, scaling curve

Desteklenen modeller:
1. lstm              — Tek yönlü LSTM (baseline)
2. bilstm            — Bidirectional LSTM
3. bilstm_att        — BiLSTM + Custom Attention
4. bilstm_selfatt    — BiLSTM + Self-Attention (SA-BiLSTM)
5. cnn_bilstm_att    — CNN + BiLSTM + Attention (hibrit)
"""

from __future__ import annotations

import re
import time
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout, Input, Conv1D, MaxPooling1D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ╔══════════════════════════════════════════════════════════════╗
# ║                     CONTROL PANEL                           ║
# ╚══════════════════════════════════════════════════════════════╝

DATA_PATH = Path(r"C:\Users\berka\AI-Information-Detection\src\lstm\data_center\combined_dataset.csv")
DATA_PATH = Path(r"D:\projects\AI-Information-Detection\src\lstm\data_center\combined_dataset.csv")

MAX_NB_WORDS = 30_000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 100  # 128 → 100: GloVe 100d ile eşleşmeli

TEST_SIZE = 0.15
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
VAL_SPLIT = 0.15
THRESHOLD = 0.5

PATIENCE_EARLY_STOP = 5
PATIENCE_LR_REDUCE = 3
LEARNING_RATE = 0.001

LSTM_UNITS = 64
DROPOUT_RATE = 0.3
CNN_FILTERS = 128
CNN_KERNEL = 3

OOV_TOKEN = "<OOV>"
TF_FILTERS = r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'

# GLOVE_PATH = r"C:\Users\berka\glove.6B.100d.txt"
GLOVE_PATH = r"D:\projects\AI-Information-Detection\src\lstm\data_center\glove.6B.100d.txt"

RESULTS_DIR = Path("results")


# ╔══════════════════════════════════════════════════════════════╗
# ║                   ATTENTION LAYERS                          ║
# ╚══════════════════════════════════════════════════════════════╝


class AttentionLayer(Layer):
    """Custom Attention (Bahdanau). IEEE Access BiLSTM makaleleri."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight("att_w", shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight("att_b", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()


class SelfAttentionLayer(Layer):
    """Self-Attention (Q-K-V). SA-BiLSTM (IEEE Access, 2024)."""

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        d = input_shape[-1]
        self.W_q = self.add_weight("q", shape=(d, self.units), initializer="glorot_uniform")
        self.W_k = self.add_weight("k", shape=(d, self.units), initializer="glorot_uniform")
        self.W_v = self.add_weight("v", shape=(d, self.units), initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, x):
        Q, K, V = tf.matmul(x, self.W_q), tf.matmul(x, self.W_k), tf.matmul(x, self.W_v)
        s = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.units, tf.float32))
        return tf.reduce_mean(tf.matmul(tf.nn.softmax(s, axis=-1), V), axis=1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# ╔══════════════════════════════════════════════════════════════╗
# ║                    DATA FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════╝


def normalize_binary_labels(y: pd.Series) -> np.ndarray:
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().all():
        uniq = sorted(set(y_num.astype(int).tolist()))
        if uniq == [0, 1]:
            return y_num.astype(np.int32).to_numpy()
    y_str = y.astype(str).str.strip().str.lower()
    uniq = sorted(y_str.unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"Expected 2 labels, got {len(uniq)}: {uniq}")
    mapping = {"human": 0, "ai": 1} if ("human" in uniq and "ai" in uniq) else {uniq[0]: 0, uniq[1]: 1}
    return y_str.map(mapping).astype(np.int32).to_numpy()


def clean_text(text: str) -> str:
    """Basit temizleme. Stop word KALDIRILMADI — Chang(2025) gösterdi ki
    AI/human farklı stop word pattern'leri gösteriyor, kaldırmak sinyal kaybettirir."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def stratified_split_indices(y, test_size, seed):
    rng = np.random.default_rng(seed)
    train_p, test_p = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = int(round(len(idx) * test_size))
        test_p.append(idx[:n])
        train_p.append(idx[n:])
    tr, te = np.concatenate(train_p), np.concatenate(test_p)
    rng.shuffle(tr)
    rng.shuffle(te)
    return tr, te


def stratified_kfold_indices(y, k, seed):
    rng = np.random.default_rng(seed)
    pcf = {}
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        pcf[c] = np.array_split(idx, k)
    splits = []
    for i in range(k):
        te = np.concatenate([pcf[c][i] for c in np.unique(y)])
        tr = np.concatenate([np.concatenate([pcf[c][j] for j in range(k) if j != i]) for c in np.unique(y)])
        rng.shuffle(te)
        rng.shuffle(tr)
        splits.append((tr, te))
    return splits


def binary_metrics(y_true, y_pred):
    y_true, y_pred = y_true.astype(int), y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-12, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


def compute_class_weights(y):
    """Wu et al. survey — dengesiz veri yönetimi. Senin: 2597 vs 1915."""
    counts = np.bincount(y)
    total = len(y)
    return {i: total / (len(counts) * c) for i, c in enumerate(counts)}


def load_and_prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    tc, lc = df.columns[0], df.columns[1]
    df = df.dropna(subset=[tc, lc]).drop_duplicates(subset=[tc, lc]).reset_index(drop=True)
    df[tc] = df[tc].apply(clean_text)
    df = df[df[tc].str.len() > 0].reset_index(drop=True)
    print(f"Dataset: {len(df)} rows | Columns: '{tc}', '{lc}'")
    print(f"Labels: {df[lc].value_counts().to_dict()}")
    wc = df[tc].str.split().str.len()
    print(f"Words/sample: mean={wc.mean():.0f}, median={wc.median():.0f}, min={wc.min()}, max={wc.max()}\n")
    return df[tc].astype(str).to_numpy(), normalize_binary_labels(df[lc])


def tokenize_and_pad(x_train_text, x_test_text):
    tok = Tokenizer(num_words=MAX_NB_WORDS, filters=TF_FILTERS, lower=True, oov_token=OOV_TOKEN)
    tok.fit_on_texts(x_train_text.tolist())
    xtr = pad_sequences(tok.texts_to_sequences(x_train_text.tolist()), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    xte = pad_sequences(tok.texts_to_sequences(x_test_text.tolist()), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    print(f"Vocab: {min(len(tok.word_index)+1, MAX_NB_WORDS)} words")
    return xtr, xte, tok


# ╔══════════════════════════════════════════════════════════════╗
# ║                   GLOVE EMBEDDING                           ║
# ╚══════════════════════════════════════════════════════════════╝


def load_glove_matrix(path, word_index, dim):
    print(f"GloVe loading: {path}")
    emb = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            emb[p[0]] = np.asarray(p[1:], dtype="float32")
    vs = min(len(word_index) + 1, MAX_NB_WORDS)
    mat = np.zeros((vs, dim))
    found = sum(1 for w, i in word_index.items() if i < vs and w in emb and not mat.__setitem__(i, emb[w]))
    # simpler count
    found = sum(1 for i in range(vs) if np.any(mat[i] != 0))
    print(f"GloVe: {found}/{vs} matched ({100*found/vs:.1f}%)")
    return mat


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MODEL BUILDERS                           ║
# ╚══════════════════════════════════════════════════════════════╝


def build_model(name: str, tokenizer=None) -> tf.keras.Model:
    glove = None
    if GLOVE_PATH and tokenizer:
        glove = load_glove_matrix(GLOVE_PATH, tokenizer.word_index, EMBEDDING_DIM)
    vs = min(len(tokenizer.word_index) + 1, MAX_NB_WORDS) if tokenizer else MAX_NB_WORDS

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    if glove is not None:
        x = Embedding(vs, EMBEDDING_DIM, weights=[glove], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inp)
    else:
        x = Embedding(vs, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(inp)
    x = SpatialDropout1D(0.2)(x)

    if name == "lstm":
        x = LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)(x)
    elif name == "bilstm":
        x = Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2))(x)
    elif name == "bilstm_att":
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
        x = AttentionLayer(name="attention")(x)
    elif name == "bilstm_selfatt":
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
        x = SelfAttentionLayer(units=LSTM_UNITS, name="self_attention")(x)
    elif name == "cnn_bilstm_att":
        x = Conv1D(CNN_FILTERS, CNN_KERNEL, activation="relu")(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
        x = AttentionLayer(name="attention")(x)
    else:
        raise ValueError(f"Unknown: {name}")

    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)

    m = Model(inp, out, name=name)
    m.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ╔══════════════════════════════════════════════════════════════╗
# ║                  TRAIN & EVALUATE                           ║
# ╚══════════════════════════════════════════════════════════════╝


def train_and_evaluate(name, x_text, y, train_idx, test_idx, verbose=1):
    tf.keras.backend.clear_session()
    xtr_t, xte_t = x_text[train_idx], x_text[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    xtr_p, xte_p, tok = tokenize_and_pad(xtr_t, xte_t)
    cw = compute_class_weights(ytr)
    model = build_model(name, tok)
    if verbose:
        print(f"Model: {name} | Params: {model.count_params():,}")

    cb = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_EARLY_STOP, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_LR_REDUCE, min_lr=1e-6, verbose=0),
    ]
    h = model.fit(xtr_p, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT, class_weight=cw, callbacks=cb, verbose=verbose)

    yp = (model.predict(xte_p, verbose=0).reshape(-1) >= THRESHOLD).astype(np.int32)
    met = binary_metrics(yte, yp)
    met["epochs_trained"] = len(h.history["loss"])
    met["final_val_acc"] = h.history["val_accuracy"][-1]
    return met


# ╔══════════════════════════════════════════════════════════════╗
# ║                    RUN MODES                                ║
# ╚══════════════════════════════════════════════════════════════╝


def run_quick_test(name):
    t0 = time.perf_counter()
    print(f"\n{'='*60}\nQUICK TEST: {name}\n{'='*60}")
    x, y = load_and_prepare_data()
    tri, tei = stratified_split_indices(y, TEST_SIZE, RANDOM_SEED)
    print(f"Train: {len(tri)}, Test: {len(tei)}")
    m = train_and_evaluate(name, x, y, tri, tei)
    print(f"\n--- {name} RESULT ---")
    for k in ["accuracy", "f1", "precision", "recall"]:
        print(f"  {k}: {m[k]:.4f}")
    print(f"  epochs: {m['epochs_trained']}, val_acc: {m['final_val_acc']:.4f}")
    print(f"  CM: TN={int(m['tn'])} FP={int(m['fp'])} | FN={int(m['fn'])} TP={int(m['tp'])}")
    print(f"  Time: {time.perf_counter()-t0:.1f}s\n")
    return m


def run_cv(name, k=5):
    t0 = time.perf_counter()
    print(f"\n{'='*60}\nCV: {name} | k={k}\n{'='*60}")
    x, y = load_and_prepare_data()
    splits = stratified_kfold_indices(y, k, RANDOM_SEED)
    ml = []
    for i, (tri, tei) in enumerate(splits, 1):
        print(f"\n--- Fold {i}/{k} ---")
        m = train_and_evaluate(name, x, y, tri, tei, verbose=0)
        print(f"  Acc={m['accuracy']:.4f} F1={m['f1']:.4f} Ep={m['epochs_trained']}")
        ml.append(m)
    avg = {}
    print(f"\n{'='*60}\n{name} CV RESULT ({k}-fold)\n{'='*60}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        v = np.array([m[key] for m in ml])
        avg[key], avg[f"{key}_std"] = float(v.mean()), float(v.std())
        print(f"  {key}: {v.mean():.4f} +/- {v.std():.4f}")
    print(f"  Time: {time.perf_counter()-t0:.1f}s\n")
    return avg


def run_comparison(use_cv=False, k=5):
    models = ["lstm", "bilstm", "bilstm_att", "bilstm_selfatt", "cnn_bilstm_att"]
    results = {}
    for n in models:
        results[n] = run_cv(n, k) if use_cv else run_quick_test(n)

    print(f"\n{'='*70}\nMODEL COMPARISON TABLE\n{'='*70}")
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)
    for n, m in results.items():
        if use_cv:
            print(
                f"{n:<20} {m['accuracy']:.4f}±{m['accuracy_std']:.3f} "
                f"{m['f1']:.4f}±{m['f1_std']:.3f} "
                f"{m['precision']:.4f}±{m['precision_std']:.3f} "
                f"{m['recall']:.4f}±{m['recall_std']:.3f}"
            )
        else:
            print(f"{n:<20} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")
    print("=" * 70)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_DIR/'comparison.json'}\n")
    return results


# ╔══════════════════════════════════════════════════════════════╗
# ║              SAVE / LOAD (eğitim sonrası kaydet)             ║
# ╚══════════════════════════════════════════════════════════════╝

import pickle


def save_model_and_tokenizer(model, tokenizer, model_name: str):
    """Model ve tokenizer'ı kaydet — sonradan predict için lazım."""
    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = RESULTS_DIR / f"{model_name}_model.h5"
    tok_path = RESULTS_DIR / f"{model_name}_tokenizer.pkl"

    model.save(str(model_path))
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Model saved: {model_path}")
    print(f"Tokenizer saved: {tok_path}")
    return model_path, tok_path


def load_model_and_tokenizer(model_name: str):
    """Kaydedilmiş model ve tokenizer'ı yükle."""
    model_path = RESULTS_DIR / f"{model_name}_model.h5"
    tok_path = RESULTS_DIR / f"{model_name}_tokenizer.pkl"

    model = tf.keras.models.load_model(str(model_path), custom_objects={"AttentionLayer": AttentionLayer, "SelfAttentionLayer": SelfAttentionLayer}, compile=False)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"Model loaded: {model_path}")
    print(f"Tokenizer loaded: {tok_path}")
    return model, tokenizer


# ╔══════════════════════════════════════════════════════════════╗
# ║         TRAIN + SAVE (en iyi modeli eğit ve kaydet)         ║
# ╚══════════════════════════════════════════════════════════════╝


def run_train_and_save(model_name: str = "bilstm_att"):
    """Modeli eğit, değerlendir VE kaydet — sonra predict için kullanılacak."""
    t0 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"TRAIN & SAVE: {model_name}")
    print(f"{'='*60}")

    x_text, y = load_and_prepare_data()
    train_idx, test_idx = stratified_split_indices(y, TEST_SIZE, RANDOM_SEED)

    xtr_t, xte_t = x_text[train_idx], x_text[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    # Tokenize
    xtr_p, xte_p, tokenizer = tokenize_and_pad(xtr_t, xte_t)
    cw = compute_class_weights(ytr)

    # Build
    model = build_model(model_name, tokenizer)
    print(f"Params: {model.count_params():,}")

    # Train
    cb = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_EARLY_STOP, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_LR_REDUCE, min_lr=1e-6, verbose=1),
    ]
    model.fit(xtr_p, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT, class_weight=cw, callbacks=cb, verbose=1)

    # Evaluate
    yp = (model.predict(xte_p, verbose=0).reshape(-1) >= THRESHOLD).astype(np.int32)
    m = binary_metrics(yte, yp)

    print(f"\n--- {model_name} RESULT ---")
    for k in ["accuracy", "f1", "precision", "recall"]:
        print(f"  {k}: {m[k]:.4f}")
    print(f"  CM: TN={int(m['tn'])} FP={int(m['fp'])} | FN={int(m['fn'])} TP={int(m['tp'])}")

    # KAYDET
    save_model_and_tokenizer(model, tokenizer, model_name)

    print(f"  Time: {time.perf_counter()-t0:.1f}s\n")
    return model, tokenizer, m


# ╔══════════════════════════════════════════════════════════════╗
# ║          PREDICT — Kendi cümleni test et                    ║
# ╚══════════════════════════════════════════════════════════════╝


def predict_single(text: str, model, tokenizer) -> dict:
    """Tek bir metin için tahmin yap."""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "AI" if prob >= THRESHOLD else "HUMAN"
    confidence = prob if prob >= THRESHOLD else 1 - prob
    return {"label": label, "confidence": f"{confidence:.1%}", "raw_score": f"{prob:.4f}"}


def interactive_predict(model_name: str = "bilstm_att"):
    """
    Kaydedilmiş modeli yükle ve interaktif tahmin yap.
    Kendi cümleni yaz, AI mı Human mı söylesin.
    Çıkmak için 'q' yaz.
    """
    print(f"\n{'='*60}")
    print(f"INTERACTIVE PREDICT MODE — {model_name}")
    print(f"{'='*60}")
    print("Bir metin yaz, model AI mi Human mı tahmin etsin.")
    print("Çıkmak için 'q' yaz.\n")

    model, tokenizer = load_model_and_tokenizer(model_name)

    while True:
        text = input("\n📝 Metin gir: ").strip()
        if text.lower() == "q":
            print("Çıkılıyor...")
            break
        if len(text) < 10:
            print("⚠ Çok kısa metin — en az birkaç cümle yaz.")
            continue

        result = predict_single(text, model, tokenizer)
        emoji = "🤖" if result["label"] == "AI" else "👤"
        print(f"\n{emoji} Tahmin: {result['label']}")
        print(f"   Güven : {result['confidence']}")
        print(f"   Skor  : {result['raw_score']} (0=Human, 1=AI)")


def batch_predict(texts: list, model_name: str = "bilstm_att"):
    """
    Birden fazla metni toplu tahmin et.
    Sunumda demo için: insan yazısı ve AI versiyonunu yan yana göster.
    """
    print(f"\n{'='*60}")
    print(f"BATCH PREDICT — {model_name}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_name)

    for i, text in enumerate(texts, 1):
        result = predict_single(text, model, tokenizer)
        preview = text[:80] + "..." if len(text) > 80 else text
        emoji = "🤖" if result["label"] == "AI" else "👤"
        print(f"\n[{i}] {emoji} {result['label']} ({result['confidence']}) | \"{preview}\"")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        MAIN                                 ║
# ╚══════════════════════════════════════════════════════════════╝


def main():
    tf.get_logger().setLevel("ERROR")
    print("=" * 60)
    print("BiLSTM Test Pipeline v2 — 5 Paper Analysis")
    print("=" * 60)

    # ╔════════════════════════════════════════════╗
    # ║  Aşağıdakilerden BİRİNİ aç, diğerlerini  ║
    # ║  yorum satırı bırak.                      ║
    # ╚════════════════════════════════════════════╝

    # --- ADIM 1: Tek model hızlı test (~1-2 dk) ---
    # run_quick_test("bilstm_att")

    # --- ADIM 2: 5 model karşılaştır, tek split (~8-12 dk) --- # BU
    run_comparison(use_cv=False)

    # --- ADIM 3: 5 model CV karşılaştır (~40-60 dk) ---
    # run_comparison(use_cv=True, k=5)

    # --- ADIM 4: En iyi modeli eğit + KAYDET ---
    # run_train_and_save("bilstm_att")

    # --- ADIM 5: Kendi cümleni test et (önce ADIM 4 çalıştır!) ---
    # interactive_predict("bilstm_att")

    # --- ADIM 6: Toplu test (sunumda demo için) ---
    """batch_predict([
        "I was walking home yesterday and saw this weird bird just sitting on the fence staring at me",
        "The implementation of advanced neural network architectures has demonstrated significant improvements in natural language processing tasks, particularly in the domain of text classification.",
    ], model_name="bilstm_att")
    """


if __name__ == "__main__":
    main()
