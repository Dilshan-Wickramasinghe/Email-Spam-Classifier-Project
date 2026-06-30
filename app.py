"""
Fake News Detection — Streamlit App
Run: streamlit run app.py
"""

import asyncio
import sys
import os
import tempfile
import time
from pathlib import Path

# Fix for Python 3.12+ on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from fake_news_detection_model import (
    clean_text,
    load_data,
    preprocess,
    MODELS,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEAT,
)

# ── Always resolve CSVs relative to THIS file, regardless of where
#    streamlit is launched from (fixes Windows working-directory issues)
_HERE     = Path(__file__).resolve().parent
TRUE_PATH = _HERE / "True.csv"
FAKE_PATH = _HERE / "Fake.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "trained": False, "models": {}, "vectorizer": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Cached training ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_training(true_path: str, fake_path: str, test_size: float, max_feat: int):
    news = preprocess(load_data(Path(true_path), Path(fake_path)))

    X, y = news["text"], news["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=max_feat, sublinear_tf=True)
    xv_train = vectorizer.fit_transform(x_train)
    xv_test  = vectorizer.transform(x_test)

    fitted = {}
    for name, model in MODELS.items():
        model.fit(xv_train, y_train)
        fitted[name] = model

    return vectorizer, fitted

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    st.subheader("Dataset Paths")
    # Show the resolved absolute paths as defaults so the user can see exactly
    # where the app is looking — and edit them if needed
    true_path = st.text_input("True News CSV", value=str(TRUE_PATH))
    fake_path = st.text_input("Fake News CSV", value=str(FAKE_PATH))

    st.markdown("**— or upload files —**")
    up_true = st.file_uploader("Upload True.csv", type="csv", key="up_true")
    up_fake = st.file_uploader("Upload Fake.csv", type="csv", key="up_fake")

    st.divider()
    st.subheader("Hyperparameters")
    test_size = st.slider("Test split ratio", 0.1, 0.4, float(TEST_SIZE), 0.05)
    max_feat  = st.select_slider(
        "TF-IDF max features",
        options=[10_000, 20_000, 30_000, 50_000, 80_000, 100_000],
        value=int(TFIDF_MAX_FEAT),
    )

    st.divider()
    train_btn = st.button("Train Models", use_container_width=True, type="primary")

# Handle uploaded files — takes priority over path inputs
_tmp_dir = tempfile.mkdtemp()
if up_true:
    tmp = os.path.join(_tmp_dir, "True.csv")
    open(tmp, "wb").write(up_true.read())
    true_path = tmp
if up_fake:
    tmp = os.path.join(_tmp_dir, "Fake.csv")
    open(tmp, "wb").write(up_fake.read())
    fake_path = tmp

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Fake News Detector")
st.caption("TF-IDF + ML · Logistic Regression · Decision Tree · Random Forest · Gradient Boosting")
st.divider()

# ── Training ──────────────────────────────────────────────────────────────────
if train_btn:
    true_p = Path(true_path)
    fake_p = Path(fake_path)

    if not true_p.exists() or not fake_p.exists():
        st.error(
            f"CSV files not found.\n\n"
            f"Looking for:\n- `{true_p}`\n- `{fake_p}`\n\n"
            f"Either correct the paths in the sidebar or use the upload buttons."
        )
        st.stop()

    with st.spinner("Training models — takes ~1–2 minutes on first run ..."):
        try:
            vectorizer, fitted = run_training(
                str(true_p), str(fake_p), test_size, max_feat
            )
            st.session_state.update({
                "trained": True, "models": fitted, "vectorizer": vectorizer,
            })
            st.success("All models trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.session_state.trained:
    st.subheader("Predict an Article")

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        article_text = st.text_area(
            "Paste a news article or headline:",
            height=220,
            placeholder="e.g. 'Scientists discover new cancer treatment ...'",
        )
        chosen_model = st.selectbox(
            "Primary model:",
            options=list(st.session_state.models.keys()),
        )
        predict_btn = st.button("Analyse Article", type="primary", use_container_width=True)

    with col_right:
        if predict_btn:
            if not article_text.strip():
                st.warning("Please paste some text first.")
            else:
                vec  = st.session_state.vectorizer
                mdls = st.session_state.models
                xv   = vec.transform([clean_text(article_text)])

                st.markdown("**All models:**")
                agree_rows = []
                for name, m in mdls.items():
                    p   = m.predict(xv)[0]
                    row = {"Model": name, "Verdict": "Fake" if p == 0 else "Real"}
                    if hasattr(m, "predict_proba"):
                        pr = m.predict_proba(xv)[0]
                        row["Confidence"] = f"{pr[p]*100:.1f}%"
                    agree_rows.append(row)
                st.dataframe(pd.DataFrame(agree_rows).set_index("Model"), use_container_width=True)
        else:
            st.info("Enter an article and click Analyse Article to see predictions.")

else:
    st.info("Click **Train Models** in the sidebar to get started.")
    st.markdown("""
    ### How it works
    1. Click **Train Models** — loads your CSVs, cleans text, trains 4 ML classifiers
    2. Paste any article or headline to get instant predictions from all models
    """)