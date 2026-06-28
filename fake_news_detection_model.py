"""
Fake News Detection - Optimized Python Script
Converted from Google Colab notebook.

Dataset expected structure:
    data/
        True.csv
        Fake.csv

Each CSV must have columns: title, text, subject, date
"""

import re
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_DIR   = Path("data")
TRUE_PATH  = DATA_DIR / "True.csv"
FAKE_PATH  = DATA_DIR / "Fake.csv"

TEST_SIZE       = 0.3
RANDOM_STATE    = 42
TFIDF_MAX_FEAT  = 50_000   # cap vocabulary; speeds up training, minimal accuracy loss


# ──────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────
_URL_RE  = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_PUNC_RE = re.compile(r"[^\w\s]")
_DIGS_RE = re.compile(r"\d+")
_NEWL_RE = re.compile(r"\n+")
_WSPC_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase + strip URLs, HTML tags, punctuation, digits, extra whitespace."""
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = _PUNC_RE.sub("", text)
    text = _DIGS_RE.sub("", text)
    text = _NEWL_RE.sub(" ", text)
    text = _WSPC_RE.sub(" ", text)
    return text.strip()


# ──────────────────────────────────────────────
# Data loading & preparation
# ──────────────────────────────────────────────
def load_data(true_path: Path, fake_path: Path) -> pd.DataFrame:
    """Load, label, merge, and shuffle the two CSV files."""
    log.info("Loading datasets …")
    true = pd.read_csv(true_path)
    fake = pd.read_csv(fake_path)

    true["label"] = 1   # Real news
    fake["label"] = 0   # Fake news

    news = (
        pd.concat([fake, true], axis=0, ignore_index=True)
        .drop(columns=["title", "subject", "date"])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    log.info(f"Total samples: {len(news):,}  |  Fake: {(news.label==0).sum():,}  |  Real: {(news.label==1).sum():,}")
    return news


def preprocess(news: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to the 'text' column (vectorised via pandas apply)."""
    log.info("Cleaning text …")
    news = news.copy()
    news["text"] = news["text"].astype(str).apply(clean_text)

    # Drop rows that became empty after cleaning
    empty_mask = news["text"].str.strip() == ""
    if empty_mask.any():
        log.warning(f"Dropping {empty_mask.sum()} empty rows after cleaning.")
        news = news[~empty_mask].reset_index(drop=True)

    return news


# ──────────────────────────────────────────────
# Model training & evaluation
# ──────────────────────────────────────────────
MODELS = {
    "Logistic Regression":       LogisticRegression(max_iter=1000, n_jobs=-1),
    "Decision Tree":             DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest":             RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting":         GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}


def train_and_evaluate(xv_train, xv_test, y_train, y_test) -> dict:
    """Train all models, print reports, return fitted model dict."""
    fitted = {}
    results = {}

    for name, model in MODELS.items():
        log.info(f"Training {name} …")
        t0 = time.perf_counter()
        model.fit(xv_train, y_train)
        elapsed = time.perf_counter() - t0

        preds = model.predict(xv_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"\n{'='*55}")
        print(f"  {name}  |  Accuracy: {acc:.4f}  |  Time: {elapsed:.1f}s")
        print(f"{'='*55}")
        print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

        fitted[name] = model

    # Summary table
    print("\n── Model Comparison ──────────────────────────────────")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 40)
        print(f"  {name:<25} {acc:.4f}  {bar}")
    print()

    return fitted


# ──────────────────────────────────────────────
# Manual / interactive testing
# ──────────────────────────────────────────────
LABEL_MAP = {0: "🔴 FAKE NEWS", 1: "🟢 REAL NEWS"}


def predict_article(text: str, vectorizer: TfidfVectorizer, models: dict) -> None:
    """Clean, vectorise, and predict one news article with all fitted models."""
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])

    print("\n── Prediction Results ────────────────────────────────")
    for name, model in models.items():
        pred = model.predict(vec)[0]
        # Show probability if the model supports it
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vec)[0][pred]
            print(f"  {name:<25}  →  {LABEL_MAP[pred]}  (confidence: {prob:.2%})")
        else:
            print(f"  {name:<25}  →  {LABEL_MAP[pred]}")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fake News Detection")
    parser.add_argument("--true",  default=str(TRUE_PATH), help="Path to True.csv")
    parser.add_argument("--fake",  default=str(FAKE_PATH), help="Path to Fake.csv")
    parser.add_argument("--test",  type=float, default=TEST_SIZE, help="Test split ratio")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive prediction mode after training")
    args = parser.parse_args()

    # ── Load & prepare data ──────────────────────────────
    news = load_data(Path(args.true), Path(args.fake))
    news = preprocess(news)

    X = news["text"]
    y = news["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"Train: {len(x_train):,}  |  Test: {len(x_test):,}")

    # ── TF-IDF vectorisation ─────────────────────────────
    log.info("Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, sublinear_tf=True)
    xv_train = vectorizer.fit_transform(x_train)
    xv_test  = vectorizer.transform(x_test)
    log.info(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # ── Train & evaluate ─────────────────────────────────
    fitted_models = train_and_evaluate(xv_train, xv_test, y_train, y_test)

    # ── Interactive prediction loop ───────────────────────
    if args.interactive:
        print("\n── Interactive Prediction Mode (type 'quit' to exit) ──")
        while True:
            try:
                article = input("\nPaste a news article or headline:\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if article.lower() in {"quit", "exit", "q"}:
                break
            if article:
                predict_article(article, vectorizer, fitted_models)


if __name__ == "__main__":
    main()