"""Paper classifier: NLTK stemming + lemmatization -> TF-IDF -> Logistic Regression.

Trained on bundled arXiv abstract samples (data/arxiv_samples.json) at first use.
"""

from __future__ import annotations

import json
import os
import string
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "data", "arxiv_samples.json")

_NLTK_RESOURCES = ("punkt_tab", "wordnet", "omw-1.4", "stopwords")


class PaperClassifier:
    """Stem + lemmatize tokenizer with TF-IDF + logistic regression."""

    def __init__(self, data_path: str = DATA_PATH):
        self._ready = False
        self.data_path = data_path
        self.pipeline: Optional[Pipeline] = None
        self.categories: list[str] = []

    # -- NLTK internals ----------------------------------------------------
    def _ensure_nltk(self) -> None:
        import nltk

        for res in _NLTK_RESOURCES:
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)

        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        self._tokenize = word_tokenize
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = set(stopwords.words("english"))

    def _preprocess(self, text: str) -> list[str]:
        toks = [t for t in self._tokenize(text.lower())
                if t not in string.punctuation and t not in self._stopwords and len(t) > 2]
        return [self._stemmer.stem(self._lemmatizer.lemmatize(t)) for t in toks]

    # -- training ----------------------------------------------------------
    def train(self) -> None:
        self._ensure_nltk()
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts, labels = [], []
        for category, samples in data.items():
            for sample in samples:
                texts.append(sample)
                labels.append(category)
        self.categories = list(data.keys())

        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(
                    tokenizer=self._preprocess,
                    preprocessor=None,
                    lowercase=False,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=1,
                )),
                ("clf", LogisticRegression(max_iter=2000, C=10.0)),
            ]
        )
        self.pipeline.fit(texts, labels)
        self.categories = list(self.pipeline.named_steps["clf"].classes_)
        self._ready = True

    def _ensure_ready(self) -> None:
        if not self._ready:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    f"Classifier training data missing: {self.data_path}"
                )
            self.train()

    # -- inference ---------------------------------------------------------
    def predict(self, text: str) -> tuple[str, float]:
        """Return (category, confidence)."""
        self._ensure_ready()
        probs = self.pipeline.predict_proba([text])[0]
        idx = int(probs.argmax())
        return self.categories[idx], float(probs[idx])

    def predict_top3(self, text: str) -> list[tuple[str, float]]:
        self._ensure_ready()
        probs = self.pipeline.predict_proba([text])[0]
        order = probs.argsort()[::-1][:3]
        return [(self.categories[i], float(probs[i])) for i in order]
