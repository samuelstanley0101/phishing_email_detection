#!/usr/bin/env python3
"""Baseline Logistic Regression models on phishing datasets."""
import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from pandas.errors import ParserError
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "logistic_results"
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset(filepath: Path) -> Tuple[pd.Series, pd.Series]:
    try:
        df = pd.read_csv(filepath)
    except ParserError:
        df = pd.read_csv(filepath, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["body", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df["body"], df["label"]


# TF-IDF Vectorizer (unigrams only)
def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 1),
        stop_words="english",
        lowercase=True,
    )


# N-gram Vectorizer (1-2 grams using CountVectorizer)
def build_ngram_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        n_jobs=1,  # single core to avoid joblib temp spills
        class_weight="balanced",
        solver="lbfgs",
    )


def evaluate_model(
    dataset_label: str,
    model_name: str,
    vectorizer_factory: Callable[[], object],
    X: pd.Series,
    y: pd.Series,
) -> Tuple[Dict[str, float | str], str]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("vectorizer", vectorizer_factory()),
        ("clf", make_classifier()),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    bacc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Safe Email (0)", "Phishing Email (1)"],
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(OUTPUT_DIR)
    cv_scores = cross_val_score(
        pipeline,
        X, # type: ignore
        y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=1,
    )

    output_text = f"\n=== {dataset_label} | {model_name} ===\n"
    output_text += f"Train size: {len(X_train)} | Test size: {len(X_test)}\n"
    output_text += f"Accuracy: {acc:.3f}\n"
    output_text += f"Balanced accuracy: {bacc:.3f}\n"
    output_text += f"CV balanced accuracy (5-fold): {np.round(cv_scores, 3)}\n"
    output_text += f"Mean CV balanced accuracy: {cv_scores.mean():.3f}\n"
    output_text += report # type: ignore

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3.5))
    plt.imshow(cm, cmap="coolwarm", aspect="auto")
    plt.title(f"Confusion Matrix – {dataset_label} | {model_name}")
    plt.xticks([0, 1], ["Pred Safe", "Pred Phish"])
    plt.yticks([0, 1], ["True Safe", "True Phish"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black", fontweight="bold")
    plt.colorbar(label="Count")
    plt.tight_layout()
    
    safe_model_name = model_name.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
    cm_filename = OUTPUT_DIR / f"cm_{dataset_label}_{safe_model_name}.png"
    plt.savefig(cm_filename, dpi=100, bbox_inches="tight")
    plt.close()

    return {
        "Dataset": dataset_label,
        "Model": model_name,
        "Accuracy": acc,
        "Balanced Accuracy": bacc,
        "CV Balanced Accuracy Mean": cv_scores.mean(),
    }, output_text


def main():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression baselines on a single dataset file."
    )
    parser.add_argument(
        "dataset_file",
        type=Path,
        help="Path to CSV with 'body' and 'label' columns (e.g., output from make-subset.py)",
    )
    args = parser.parse_args()

    if not args.dataset_file.exists():
        print(f"Error: Dataset file not found: {args.dataset_file}")
        return

    dataset_label = args.dataset_file.stem
    print(f"Loading dataset from {args.dataset_file}...")
    X, y = load_dataset(args.dataset_file)
    print(f"Loaded {len(X)} samples\n")

    results: List[Dict[str, float]] = []
    all_output = ""

    result1, output1 = evaluate_model(
        dataset_label,
        "LogReg + TF-IDF (uni)",
        build_tfidf_vectorizer,
        X,
        y,
    )
    results.append(result1)
    all_output += output1

    result2, output2 = evaluate_model(
        dataset_label,
        "LogReg + Count (1-2gram)",
        build_ngram_vectorizer,
        X,
        y,
    )
    results.append(result2)
    all_output += output2

    summary_df = pd.DataFrame(results)
    summary_text = "\n===== Summary =====\n" + summary_df.to_string()
    all_output += summary_text
    
    # Save to file
    results_file = OUTPUT_DIR / "results.txt"
    with open(results_file, "w") as f:
        f.write(all_output)
    
    print(f"Results saved to {results_file}")
    print(all_output)


if __name__ == "__main__":
    main()
