"""Paper classifier: TF-IDF + Logistic Regression on arXiv abstracts.

Cached to disk after first training so subsequent loads are instant.
"""

from __future__ import annotations

import json
import os
import pickle
import string
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "data", "arxiv_samples.json")
CACHE_PATH = os.path.join(_BASE_DIR, "data", "classifier_cache.pkl")


class PaperClassifier:
    """TF-IDF + logistic regression classifier with disk cache."""

    def __init__(self, data_path: str = DATA_PATH):
        self._ready = False
        self.data_path = data_path
        self.pipeline: Optional[Pipeline] = None
        self.categories: list[str] = []

    def _load_cache(self) -> bool:
        if not os.path.exists(CACHE_PATH):
            return False
        try:
            with open(CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            self.pipeline = data["pipeline"]
            self.categories = data["categories"]
            self._ready = True
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump({"pipeline": self.pipeline, "categories": self.categories}, f)
        except Exception:
            pass

    def train(self) -> None:
        import nltk
        for res in ("punkt_tab", "wordnet", "omw-1.4", "stopwords"):
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)

        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        def tokenize(text):
            return [
                stemmer.stem(lemmatizer.lemmatize(t))
                for t in word_tokenize(text.lower())
                if t not in string.punctuation and t not in stop_words and len(t) > 2
            ]

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts, labels = [], []
        for category, samples in data.items():
            for sample in samples:
                texts.append(sample)
                labels.append(category)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                tokenizer=tokenize, lowercase=False, token_pattern=None,
                ngram_range=(1, 2), min_df=1,
            )),
            ("clf", LogisticRegression(max_iter=2000, C=10.0)),
        ])
        self.pipeline.fit(texts, labels)
        self.categories = list(self.pipeline.named_steps["clf"].classes_)
        self._ready = True
        self._save_cache()

    def _ensure_ready(self) -> None:
        if not self._ready:
            if self._load_cache():
                return
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Classifier data missing: {self.data_path}")
            self.train()

    def predict(self, text: str) -> tuple[str, float]:
        self._ensure_ready()
        probs = self.pipeline.predict_proba([text])[0]
        idx = int(probs.argmax())
        return self.categories[idx], float(probs[idx])

    def predict_top3(self, text: str) -> list[tuple[str, float]]:
        self._ensure_ready()
        probs = self.pipeline.predict_proba([text])[0]
        order = probs.argsort()[::-1][:3]
        return [(self.categories[i], float(probs[i])) for i in order]
