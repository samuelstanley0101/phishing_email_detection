#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB


results_summary = []


def run_balanced_nb(name: str, filepath: Path):
    print(f"\n========== {name} ==========")

    # --- Load dataset, robust to parsing issues ---
    try:
        df = pd.read_csv(filepath)
    except ParserError:
        print(f"⚠ Parsing issue in {name}, switching to python engine and skipping bad lines...")
        df = pd.read_csv(
            filepath,
            engine="python",
            on_bad_lines="skip",
        )

    # Require body + label, coerce label to int
    df = df.dropna(subset=["body", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["Label"] = df["label"].astype(int)

    print("Shape:", df.shape)
    print(df["Label"].value_counts().rename(index={0: "Safe (0)", 1: "Phish (1)"}), "\n")

    X = df["body"].values
    y = df["Label"].values

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- TF-IDF + Naive Bayes ---
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t = tfidf.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_t, y_train)
    y_pred = nb.predict(X_test_t)

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    print("=== Test Evaluation (Standard NB) ===")
    print(f"Accuracy:          {acc:.3f}")
    print(f"Balanced Accuracy: {bacc:.3f}\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Safe Email (0)", "Phishing Email (1)"],
        )
    )

    # --- Feature introspection ---
    try:
        feature_names = np.array(tfidf.get_feature_names_out())
        log_prob_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
        top_n = 10
        top_phish_idx = np.argsort(log_prob_diff)[-top_n:][::-1]
        top_safe_idx = np.argsort(log_prob_diff)[:top_n]
        print("Top indicative phishing tokens:", ", ".join(feature_names[top_phish_idx]))
        print("Top indicative safe tokens:", ", ".join(feature_names[top_safe_idx]))
    except Exception as e:
        print(f"(Feature introspection unavailable: {e})")

    # --- Confusion matrix (shown interactively as before) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(3.4, 3))
    plt.imshow(cm, cmap="coolwarm")
    plt.title(f"Confusion Matrix – {name}")
    plt.xticks([0, 1], ["Pred Safe", "Pred Phish"])
    plt.yticks([0, 1], ["True Safe", "True Phish"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()

    # --- 5-fold CV on full dataset ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all = tfidf.fit_transform(X)  # TF-IDF on all samples for CV
    nb_cv = MultinomialNB()
    bacc_cv = cross_val_score(nb_cv, X_all, y, cv=cv, scoring="balanced_accuracy")

    print("=== 5-fold CV (Balanced Accuracy) ===")
    print("Fold scores:", np.round(bacc_cv, 3))
    print("Mean Balanced Accuracy:", np.round(bacc_cv.mean(), 3))

    results_summary.append(
        {
            "Dataset": name,
            "Test Accuracy": acc,
            "Test Balanced Accuracy": bacc,
            "CV Balanced Accuracy Mean": bacc_cv.mean(),
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Naive Bayes baseline on a phishing dataset CSV (requires 'body' and 'label' columns)."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to CSV file with 'body' and 'label' columns.",
    )
    args = parser.parse_args()

    # Validate file
    if not args.dataset:
        raise ValueError("Error: no dataset supplied. You must list a dataset CSV file.")
    if not os.path.isfile(args.dataset):
        raise ValueError(f"Error: file {args.dataset} does not exist. Are you in the right directory?")

    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.name

    print(f"Loading dataset from {dataset_path}...")
    run_balanced_nb(dataset_name, dataset_path)

    # Print overall summary (single row, but keeps the old behavior)
    if results_summary:
        print("\n\n===== Overall Comparison =====")
        comp_df = pd.DataFrame(results_summary)
        print(comp_df)


if __name__ == "__main__":
    main()
