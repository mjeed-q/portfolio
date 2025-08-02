# Sentiment Analysis on Movie Reviews

A text-classification project that predicts whether a movie review is **positive** or **negative** using TF‑IDF features and a **Multinomial Naive Bayes** classifier. The notebook also includes exploratory visualizations such as class distribution and word clouds for positive vs. negative reviews.

## 📁 Contents
- `تحليل الافلام (1).ipynb` — main notebook
- (Optional) `IMDB Dataset.csv` — dataset of reviews and labels (`review`, `sentiment`)

## 🧰 Requirements
- Python 3.9+
- pandas, scikit-learn, nltk, matplotlib, seaborn, wordcloud

Install everything with:
```bash
pip install pandas scikit-learn nltk matplotlib seaborn wordcloud
```

> **NLTK data**: the notebook downloads `stopwords` on first run. If needed, add:
```python
import nltk
nltk.download("stopwords")
```

## 🧪 Approach
1. Load the IMDb dataset and split into train/test (`review` → text, `sentiment` → target).
2. Vectorize text using **TF‑IDF** with English stopwords (NLTK).
3. Train **MultinomialNB** and evaluate on the test set.
4. Visualize class balance and generate **WordClouds** for positive/negative reviews.

## ▶️ How to Run
1. Place `IMDB Dataset.csv` next to the notebook (or update the path in the code).
2. Open the notebook in Jupyter/Colab and run all cells.

## 🧯 Reproducibility
- Fixed `random_state=42` for train/test split.
- The pipeline is deterministic given the same data and stopwords list.

## 📊 Example Outputs
- Test **accuracy** printed in the notebook.
- **WordCloud** images contrasting positive vs. negative tokens.
- A count plot showing the distribution of labels.

## 🚀 Possible Extensions
- Add **SVM** or **Logistic Regression** baselines.
- Use **n‑grams** and **class weights**.
- Perform **hyperparameter tuning** with `GridSearchCV`.
- Try **spaCy** or **transformers** for richer features.

## 📄 License & Data Source
- Dataset: IMDb movie reviews (common public dataset; ensure you comply with the source license you use).
- Code: MIT (or adapt to your repository policy).
