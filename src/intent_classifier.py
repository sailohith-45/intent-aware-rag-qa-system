"""
Intent Classifier Component: Sentence Transformers + Logistic Regression.
Classifies user questions into structured intent categories to guide retrieval and answer styling.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ARTIFACTS_DIR, EMBED_MODEL, INTENT_MAP, TEST_CSV, TRAIN_CSV


def clean_text(text: str) -> str:
    """Normalize input text by lowercasing and collapsing whitespace."""
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


class IntentClassifier:
    """Intent classifier using dense sentence embeddings and Logistic Regression."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformer] = None,
        model_path: Optional[Union[str, Path]] = None,
        class_weight: Optional[str] = None,
    ):
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        self.class_weight = class_weight
        self.clf = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=42)
        self.classes_: List[str] = []
        self.is_fitted = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    @staticmethod
    def load_dataset(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
        """Load and clean questions and intent labels from a CSV file."""
        df = pd.read_csv(path)
        if "question" not in df.columns or "intent" not in df.columns:
            raise ValueError(f"CSV at {path} must contain 'question' and 'intent' columns.")
        df = df.dropna(subset=["question", "intent"])
        questions = [clean_text(q) for q in df["question"]]
        intents = df["intent"].astype(str).tolist()
        return questions, intents

    def fit(
        self,
        X_train: List[str],
        y_train: List[str],
        show_progress: bool = True,
    ) -> "IntentClassifier":
        """Encode training questions and fit the logistic regression model."""
        clean_questions = [clean_text(q) for q in X_train]
        embeddings = self.embedder.encode(
            clean_questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        self.clf.fit(embeddings, y_train)
        self.classes_ = list(self.clf.classes_)
        self.is_fitted = True
        return self

    def evaluate(
        self,
        X_test: List[str],
        y_test: List[str],
    ) -> Dict[str, Union[float, Dict, np.ndarray]]:
        """Evaluate classifier performance on held-out test data."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before evaluation.")

        clean_questions = [clean_text(q) for q in X_test]
        embeddings = self.embedder.encode(
            clean_questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        predictions = self.clf.predict(embeddings)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=self.classes_),
            "classes": self.classes_,
        }
        return metrics

    def predict(self, question: str) -> str:
        """Predict the mapped readable intent label for a question."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted or loaded before predicting.")

        embedding = self.embedder.encode(
            [clean_text(question)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        raw_label = self.clf.predict(embedding)[0]
        return INTENT_MAP.get(raw_label, raw_label)

    def predict_with_confidence(self, question: str) -> Tuple[str, float]:
        """Predict intent label alongside the model's posterior confidence."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted or loaded before predicting.")

        embedding = self.embedder.encode(
            [clean_text(question)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        proba = self.clf.predict_proba(embedding)[0]
        max_idx = int(np.argmax(proba))
        raw_label = self.clf.classes_[max_idx]
        confidence = float(proba[max_idx])
        return INTENT_MAP.get(raw_label, raw_label), confidence

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist fitted logistic regression model and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted classifier.")

        save_path = Path(path) if path else ARTIFACTS_DIR / "intent_classifier.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.clf,
            "classes": self.classes_,
            "class_weight": self.class_weight,
            "intent_map": INTENT_MAP,
        }
        joblib.dump(artifact, save_path)
        return save_path

    def load(self, path: Optional[Union[str, Path]] = None) -> "IntentClassifier":
        """Load a persisted classifier and metadata from disk."""
        load_path = Path(path) if path else ARTIFACTS_DIR / "intent_classifier.joblib"
        if not load_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {load_path}")

        artifact = joblib.load(load_path)
        self.clf = artifact["model"]
        self.classes_ = artifact.get("classes", list(self.clf.classes_))
        self.class_weight = artifact.get("class_weight")
        self.is_fitted = True
        return self
