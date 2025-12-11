import pandas as pd
import numpy as np
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)

from pandas.errors import ParserError
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 0) Define your datasets (relative to this script)
# -------------------------------------------------------------------
BASE = Path.cwd()              # current working directory
DATASET_DIR = BASE / "source-datasets" # folder that holds your CSVs

dataset_files = {
    "Assassin": DATASET_DIR / "Assassin.csv",
    "Enron": DATASET_DIR / "Enron.csv",
    "Ling": DATASET_DIR / "Ling.csv",
    "CEAS-08": DATASET_DIR / "CEAS-08.csv",
    "TREC-05": DATASET_DIR / "TREC-05.csv",
    "TREC-06": DATASET_DIR / "TREC-06.csv",
    "TREC-07": DATASET_DIR / "TREC-07.csv",
}

results_summary = []


def run_balanced_nb(name: str, filepath: Path):
    print(f"\n========== {name} ==========")

    # --- Load dataset with robust CSV handling ---
    try:
        df = pd.read_csv(filepath)
    except ParserError:
        print(f"⚠ Parsing issue in {name}, switching to python engine and skipping bad lines...")
        df = pd.read_csv(
            filepath,
            engine="python",
            on_bad_lines="skip"  # skip malformed/broken lines
        )

    # Your files use: 'body' for text, 'label' for class (0/1)
    df = df.dropna(subset=["body"]).copy()
    df["Label"] = df["label"].astype(int)

    print("Shape:", df.shape)
    print(df["Label"].value_counts().rename(index={0: "Safe (0)", 1: "Phish (1)"}), "\n")

    X = df["body"].values
    y = df["Label"].values

    # ------------------------------------------------------
    # 1) Train/test split (stratified to keep class ratio)
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------
    # 2) TF-IDF vectorization
    # ------------------------------------------------------
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t = tfidf.transform(X_test)

    # ------------------------------------------------------
    # 3) Standard Multinomial Naive Bayes
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # 4) Confusion matrix (per dataset)
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # 5) 5-fold cross-validation with balanced accuracy
    # ------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all = tfidf.fit_transform(X)  # TF-IDF on all samples for CV
    nb_cv = MultinomialNB()
    bacc_cv = cross_val_score(nb_cv, X_all, y, cv=cv, scoring="balanced_accuracy")

    print("=== 5-fold CV (Balanced Accuracy) ===")
    print("Fold scores:", np.round(bacc_cv, 3))
    print("Mean Balanced Accuracy:", np.round(bacc_cv.mean(), 3))

    # ------------------------------------------------------
    # 6) Store summary for comparison across sets
    # ------------------------------------------------------
    results_summary.append(
        {
            "Dataset": name,
            "Test Accuracy": acc,
            "Test Balanced Accuracy": bacc,
            "CV Balanced Accuracy Mean": bacc_cv.mean(),
        }
    )


# -------------------------------------------------------------------
# Run on all datasets
# -------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            filepath = BASE / filename
            run_balanced_nb(filename, filepath)
    else:
        for name, path in dataset_files.items():
            run_balanced_nb(name, path)

    print("\n\n===== Overall Comparison =====")
    comp_df = pd.DataFrame(results_summary)
    print(comp_df)
