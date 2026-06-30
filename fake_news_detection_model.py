"""
Fake News Detection - Core Model Logic
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# CSVs sit in the same folder as this file (project root)
_HERE      = Path(__file__).parent
TRUE_PATH  = _HERE / "True.csv"
FAKE_PATH  = _HERE / "Fake.csv"

TEST_SIZE      = 0.3
RANDOM_STATE   = 42
TFIDF_MAX_FEAT = 50_000

# ── Text cleaning ─────────────────────────────────────────────────────────────
_URL_RE  = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_PUNC_RE = re.compile(r"[^\w\s]")
_DIGS_RE = re.compile(r"\d+")
_WSPC_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = _PUNC_RE.sub("", text)
    text = _DIGS_RE.sub("", text)
    text = _WSPC_RE.sub(" ", text)
    return text.strip()

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(true_path: Path, fake_path: Path) -> pd.DataFrame:
    log.info("Loading datasets …")
    true = pd.read_csv(true_path)
    fake = pd.read_csv(fake_path)
    true["label"] = 1
    fake["label"] = 0
    news = (
        pd.concat([fake, true], axis=0, ignore_index=True)
        .drop(columns=["title", "subject", "date"])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    log.info(f"Total: {len(news):,}  Fake: {(news.label==0).sum():,}  Real: {(news.label==1).sum():,}")
    return news

def preprocess(news: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning text …")
    news = news.copy()
    news["text"] = news["text"].astype(str).apply(clean_text)
    empty = news["text"].str.strip() == ""
    if empty.any():
        log.warning(f"Dropping {empty.sum()} empty rows after cleaning.")
        news = news[~empty].reset_index(drop=True)
    return news

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

def train_and_evaluate(xv_train, xv_test, y_train, y_test) -> dict:
    fitted, results = {}, {}
    for name, model in MODELS.items():
        log.info(f"Training {name} …")
        t0 = time.perf_counter()
        model.fit(xv_train, y_train)
        elapsed = time.perf_counter() - t0
        preds = model.predict(xv_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"\n{'='*55}\n  {name}  |  Accuracy: {acc:.4f}  |  Time: {elapsed:.1f}s\n{'='*55}")
        print(classification_report(y_test, preds, target_names=["Fake", "Real"]))
        fitted[name] = model
    print("\n── Model Comparison ──────────────────────────────────")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} {acc:.4f}  {'█' * int(acc * 40)}")
    return fitted

# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fake News Detection")
    parser.add_argument("--true",        default=str(TRUE_PATH))
    parser.add_argument("--fake",        default=str(FAKE_PATH))
    parser.add_argument("--test",        type=float, default=TEST_SIZE)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    news = preprocess(load_data(Path(args.true), Path(args.fake)))
    X, y = news["text"], news["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test, random_state=RANDOM_STATE, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, sublinear_tf=True)
    fitted_models = train_and_evaluate(
        vectorizer.fit_transform(x_train), vectorizer.transform(x_test), y_train, y_test
    )
    if args.interactive:
        while True:
            try:
                article = input("\nPaste article (or 'quit'):\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if article.lower() in {"quit", "exit", "q"}:
                break
            if article:
                xv = vectorizer.transform([clean_text(article)])
                for name, m in fitted_models.items():
                    pred = m.predict(xv)[0]
                    label = " FAKE" if pred == 0 else " REAL"
                    print(f"  {name:<25} → {label}")

if __name__ == "__main__":
    main()