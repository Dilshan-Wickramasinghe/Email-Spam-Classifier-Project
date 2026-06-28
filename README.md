#Fake-News-Detection-using-Machine-Learning

An end-to-end machine learning system that classifies news articles as **real or fake** using NLP and classical ML classifiers. Trained on 44,898 labelled articles and deployed as an interactive **Streamlit web app** with real-time predictions and model comparison dashboards.

---

##  App Preview

| Dashboard | Prediction |
|-----------|------------|
| Model accuracy bar chart, F1 scores, train time per classifier | Paste any article → instant verdict with confidence score from all 4 models |

---

##  Features

- **Text preprocessing pipeline** — lowercasing, URL/HTML stripping, punctuation and digit removal, whitespace normalisation
- **TF-IDF vectorisation** with 50,000-feature optimised vocabulary and sublinear term-frequency scaling
- **4 ML classifiers** trained and benchmarked side by side:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- **Stratified train/test split** (70/30) for balanced class evaluation
- **Full classification reports** — precision, recall, and F1-score per class per model
- **Streamlit web app** with real-time article prediction, per-model confidence scores, and an interactive metrics dashboard
- **File upload support** — drag and drop your own CSVs directly in the sidebar

---

##  Model Performance

| Model | Accuracy | Fake F1 | Real F1 |
|-------|----------|---------|---------|
| Gradient Boosting | ~99.6% | ~1.00 | ~1.00 |
| Decision Tree | ~99.6% | ~1.00 | ~1.00 |
| Logistic Regression | ~98.9% | 0.99 | 0.99 |
| Random Forest | ~98.8% | 0.99 | 0.99 |

> Results may vary slightly across runs due to random shuffling. Evaluated on a 13,470-article held-out test set.

---

##  Dataset

| Split | Count | Label |
|-------|-------|-------|
| Real news (`True.csv`) | 21,417 | `1` |
| Fake news (`Fake.csv`) | 23,481 | `0` |
| **Total** | **44,898** | — |

**Columns:** `title`, `text`, `subject`, `date`

**Subjects in real news:** `politicsNews`, `worldnews`

**Subjects in fake news:** `News`, `politics`, `Government News`, `left-news`, `US_News`, `Middle-east`

> The dataset covers US political news articles from 2016–2017. Source: [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

##  Project Structure

```
Fake-News-Detection/
├── app.py                        # Streamlit web app
├── fake_news_detection_model.py  # Core ML pipeline (preprocessing, training, CLI)
├── True.csv                      # Real news articles
├── Fake.csv                      # Fake news articles
└── README.md
```

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Click **Train Models** in the sidebar to start.

### 4. (Optional) Run as a CLI script

```bash
python fake_news_detection_model.py

# With interactive prediction mode
python fake_news_detection_model.py --interactive

# Custom paths and split ratio
python fake_news_detection_model.py --true path/to/True.csv --fake path/to/Fake.csv --test 0.2
```

---

##  Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| ML / NLP | Scikit-learn, TF-IDF |
| Data | Pandas, NumPy |
| Web App | Streamlit |
| Preprocessing | Regex, custom pipeline |

---

##  Configuration

Key parameters are set at the top of `fake_news_detection_model.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TFIDF_MAX_FEAT` | `50,000` | TF-IDF vocabulary cap |
| `TEST_SIZE` | `0.3` | Fraction of data used for testing |
| `RANDOM_STATE` | `42` | Seed for reproducibility |

All parameters can also be adjusted interactively via the Streamlit sidebar sliders.

---

##  Sample Test Cases

**Real news (expect Real):**
> `WASHINGTON (Reuters) - The Senate voted 60-40 on Thursday to advance a bipartisan bill aimed at strengthening U.S. semiconductor manufacturing.`

**Fake news (expect Fake):**
> `BREAKING: Secret documents leaked by a whistleblower reveal that senior FBI officials conspired to frame an innocent man. The mainstream media is completely ignoring this bombshell story.`

---

##  Windows Notes

If you're on **Python 3.12+** and see `RuntimeError: Event loop is closed` or `[WinError 10054]` on shutdown, these are harmless asyncio/Windows socket errors triggered when the browser disconnects. They are suppressed in `app.py` via:

```python
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

---

##  License

This project is open source under the [MIT License](LICENSE).

---

##  Author

**Dilshan Wickramasinghe**
AI Engineer Intern · LLMs · RAG · NLP · MLOps

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/your-username)
