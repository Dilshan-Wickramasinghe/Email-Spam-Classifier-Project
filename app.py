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
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .main-title   { font-size: 2.4rem; font-weight: 700; margin-bottom: 0; }
    .sub-title    { color: #6b7280; font-size: 1rem; margin-top: 0; margin-bottom: 2rem; }
    .metric-card  { background: #f9fafb; border: 1px solid #e5e7eb;
                    border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
    .metric-val   { font-size: 1.8rem; font-weight: 700; }
    .metric-lbl   { font-size: 0.8rem; color: #6b7280; margin-top: 2px; }
    .result-fake  { background: #fef2f2; border-left: 5px solid #ef4444;
                    border-radius: 8px; padding: 1rem 1.5rem; }
    .result-real  { background: #f0fdf4; border-left: 5px solid #22c55e;
                    border-radius: 8px; padding: 1rem 1.5rem; }
    .result-title { font-size: 1.5rem; font-weight: 700; margin: 0; }
    .conf-bar-bg  { background: #e5e7eb; border-radius: 999px;
                    height: 10px; width: 100%; margin-top: 6px; }
    .conf-bar-fill{ height: 10px; border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "trained": False, "models": {}, "vectorizer": None,
    "metrics": {}, "reports": {}, "data_stats": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Cached training ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_training(true_path: str, fake_path: str, test_size: float, max_feat: int):
    news = preprocess(load_data(Path(true_path), Path(fake_path)))

    data_stats = {
        "total": len(news),
        "fake":  int((news.label == 0).sum()),
        "real":  int((news.label == 1).sum()),
    }

    X, y = news["text"], news["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=max_feat, sublinear_tf=True)
    xv_train = vectorizer.fit_transform(x_train)
    xv_test  = vectorizer.transform(x_test)

    fitted, metrics, reports = {}, {}, {}
    for name, model in MODELS.items():
        t0 = time.perf_counter()
        model.fit(xv_train, y_train)
        elapsed = time.perf_counter() - t0
        preds = model.predict(xv_test)
        fitted[name]  = model
        metrics[name] = {"accuracy": accuracy_score(y_test, preds), "time": elapsed}
        reports[name] = classification_report(
            y_test, preds, target_names=["Fake", "Real"], output_dict=True
        )

    return vectorizer, fitted, metrics, reports, data_stats

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

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
    train_btn = st.button("🚀 Train Models", use_container_width=True, type="primary")

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
st.markdown('<p class="main-title">🔍 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">TF-IDF + ML · Logistic Regression · Decision Tree · '
    'Random Forest · Gradient Boosting</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Training ──────────────────────────────────────────────────────────────────
if train_btn:
    true_p = Path(true_path)
    fake_p = Path(fake_path)

    if not true_p.exists() or not fake_p.exists():
        st.error(
            f"❌ CSV files not found.\n\n"
            f"Looking for:\n- `{true_p}`\n- `{fake_p}`\n\n"
            f"Either correct the paths in the sidebar or use the upload buttons."
        )
        st.stop()

    with st.spinner("Training models — takes ~1–2 minutes on first run …"):
        try:
            vectorizer, fitted, metrics, reports, data_stats = run_training(
                str(true_p), str(fake_p), test_size, max_feat
            )
            st.session_state.update({
                "trained": True, "models": fitted, "vectorizer": vectorizer,
                "metrics": metrics, "reports": reports, "data_stats": data_stats,
            })
            st.success("✅ All models trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

# ── Dashboard ─────────────────────────────────────────────────────────────────
if st.session_state.trained:

    st.subheader("📊 Dataset Overview")
    ds = st.session_state.data_stats
    c1, c2, c3 = st.columns(3)
    for col, label, val, color in [
        (c1, "Total Articles", f"{ds['total']:,}", "#3b82f6"),
        (c2, "Fake Articles",  f"{ds['fake']:,}",  "#ef4444"),
        (c3, "Real Articles",  f"{ds['real']:,}",  "#22c55e"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val" style="color:{color}">{val}</div>'
            f'<div class="metric-lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🏆 Model Performance")
    metrics = st.session_state.metrics
    reports = st.session_state.reports

    acc_df = pd.DataFrame(
        [(n, v["accuracy"] * 100) for n, v in metrics.items()],
        columns=["Model", "Accuracy (%)"],
    ).sort_values("Accuracy (%)", ascending=False)
    st.bar_chart(acc_df.set_index("Model")["Accuracy (%)"], height=250)

    rows = []
    for name, m in metrics.items():
        r = reports[name]
        rows.append({
            "Model":      name,
            "Accuracy":   f"{m['accuracy']:.4f}",
            "Fake F1":    f"{r['Fake']['f1-score']:.4f}",
            "Real F1":    f"{r['Real']['f1-score']:.4f}",
            "Train Time": f"{m['time']:.1f}s",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    with st.expander("📋 Detailed Classification Reports"):
        for tab, (name, report) in zip(st.tabs(list(reports.keys())), reports.items()):
            with tab:
                report_rows = []
                for cls in ["Fake", "Real", "macro avg", "weighted avg"]:
                    if cls in report:
                        r = report[cls]
                        report_rows.append({
                            "Class":     cls,
                            "Precision": f"{r['precision']:.4f}",
                            "Recall":    f"{r['recall']:.4f}",
                            "F1-Score":  f"{r['f1-score']:.4f}",
                            "Support":   int(r.get("support", 0)),
                        })
                st.dataframe(pd.DataFrame(report_rows).set_index("Class"), use_container_width=True)

    st.divider()
    st.subheader("🔮 Predict an Article")
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        article_text = st.text_area(
            "Paste a news article or headline:",
            height=220,
            placeholder="e.g. 'Scientists discover new cancer treatment …'",
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
                vec   = st.session_state.vectorizer
                mdls  = st.session_state.models
                xv    = vec.transform([clean_text(article_text)])
                model = mdls[chosen_model]
                pred  = model.predict(xv)[0]

                is_fake   = pred == 0
                css_cls   = "result-fake" if is_fake else "result-real"
                icon      = "🔴" if is_fake else "🟢"
                verdict   = "FAKE NEWS" if is_fake else "REAL NEWS"
                conf_html = ""

                if hasattr(model, "predict_proba"):
                    proba     = model.predict_proba(xv)[0]
                    conf_pct  = proba[pred] * 100
                    bar_color = "#ef4444" if is_fake else "#22c55e"
                    conf_html = (
                        f'<div class="conf-bar-bg">'
                        f'<div class="conf-bar-fill" style="width:{conf_pct:.1f}%;background:{bar_color}"></div>'
                        f'</div>'
                        f'<p style="margin:4px 0 0;font-size:0.85rem;color:#6b7280">Confidence: {conf_pct:.1f}%</p>'
                    )

                st.markdown(
                    f'<div class="{css_cls}">'
                    f'<p class="result-title">{icon} {verdict}</p>'
                    f'<p style="margin:4px 0;font-size:0.85rem;color:#6b7280">via {chosen_model}</p>'
                    f'{conf_html}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**All models:**")
                agree_rows = []
                for name, m in mdls.items():
                    p   = m.predict(xv)[0]
                    row = {"Model": name, "Verdict": "🔴 Fake" if p == 0 else "🟢 Real"}
                    if hasattr(m, "predict_proba"):
                        pr = m.predict_proba(xv)[0]
                        row["Confidence"] = f"{pr[p]*100:.1f}%"
                    agree_rows.append(row)
                st.dataframe(pd.DataFrame(agree_rows).set_index("Model"), use_container_width=True)
        else:
            st.info("Enter an article and click **Analyse Article** to see predictions.")

else:
    st.info("👈 Click **Train Models** in the sidebar to get started.")
    st.markdown("""
    ### How it works
    1. Click **Train Models** — loads your CSVs, cleans text, trains 4 ML classifiers
    2. Review accuracy metrics and F1 scores per model
    3. Paste any article or headline to get instant predictions from all models
    """)