from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from utility import TimeTracker


csv_path: str = "data/AI_Human/AI_Human_cleaned.csv"
test_size: float = 0.2
random_state: int = 0

tfidf_max_features: int = 20000
tfidf_ngram_range: Tuple[int, int] = (1, 2)
tfidf_stop_words: str = "english"

lr_max_iter: int = 1000

kfold_splits: int = 5
kfold_scoring: str = "accuracy"


def load_and_clean_dataset(csv_path: str) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns (text, label). Got {df.shape[1]}.")

    text_col, label_col = df.columns[0], df.columns[1]

    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)

    return df, text_col, label_col


def build_pipeline() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            stop_words=tfidf_stop_words,
        ),
        LogisticRegression(max_iter=lr_max_iter),
    )


def run_kfold_cv( pipeline: Pipeline, x: Any, y: Any, splits: int, random_state: int, scoring: str,) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, x, y, cv=cv, scoring=scoring)
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def split_dataset( df: pd.DataFrame, text_col: str, label_col: str, test_size: float, random_state: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    
    x_train, x_test, y_train, y_test = train_test_split(
        df[text_col].astype(str),
        df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return x_train, x_test, y_train, y_test


def overlap_count(train_text: pd.Series, test_text: pd.Series) -> int: 
    return len(set(train_text) & set(test_text))


def train_and_evaluate(pipeline: Pipeline, x_train: pd.Series, y_train: pd.Series, x_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }


def main() -> None:
    timer = TimeTracker()

    df, text_col, label_col = load_and_clean_dataset(csv_path)

    x_all = df[text_col].astype(str).values
    y_all = df[label_col].values

    pipeline = build_pipeline(cfg)

    cv_res = run_kfold_cv(
        pipeline=pipeline,
        x=x_all,
        y=y_all,
        splits=kfold_splits,
        random_state=random_state,
        scoring=kfold_scoring,
    )
    print("KFOLD")
    print("CV accuracies:", cv_res["scores"])
    print("CV mean accuracy:", cv_res["mean"])
    print("CV std:", cv_res["std"])
    print("-" * 50)

    x_train, x_test, y_train, y_test = split_dataset(
        df=df,
        text_col=text_col,
        label_col=label_col,
        test_size=test_size,
        random_state=random_state,
    )

    print("Train/Test text overlap:", overlap_count(x_train, x_test))

    eval_res = train_and_evaluate(
        pipeline=pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print("PREDICT")
    print("Accuracy:", eval_res["accuracy"])
    print("Confusion matrix:\n", eval_res["confusion_matrix"])
    print("Report:\n", eval_res["classification_report"])
    print("Total time:", timer.get_elapsed_time())


if __name__ == "__main__":
    main()
