"""
Model definitions and training utilities for Neuro Support AI.

Includes:
- TicketTypeClassifier: classify tickets as Correctivo/Evolutivo
- ChurnPredictor: regress churn_risk in [0, 100]
- ModelTrainer: helper to train and persist both models from the synthetic CSV.

This module is purposely self-contained so it can be reused both from notebooks
and from CLI scripts (see train_models.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Iterable, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ------------------------
# Low-level model wrappers
# ------------------------


class TicketTypeClassifier:
    """
    Wrapper around a text classification pipeline.

    The underlying model is a TF-IDF + LogisticRegression classifier.
    It is trained to distinguish between "Correctivo" y "Evolutivo".
    """

    def __init__(self, pipeline: Optional[Pipeline] = None):
        self.pipeline: Optional[Pipeline] = pipeline
        self.is_trained: bool = pipeline is not None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    # --- API used during training ---

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "TicketTypeClassifier":
        texts = list(texts)
        labels = list(labels)
        if not texts:
            raise ValueError("No training data provided to TicketTypeClassifier.fit()")

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        return self

    # --- API used at inference time (dashboard) ---

    def predict(self, text: Union[str, Iterable[str]]) -> Union[str, np.ndarray]:
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("TicketTypeClassifier is not trained/loaded.")

        # Aceptar tanto un solo string como una lista/serie
        if isinstance(text, str):
            preds = self.pipeline.predict([text])
            return preds[0]
        else:
            return self.pipeline.predict(list(text))

    # --- Persistence helpers ---

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save an uninitialized classifier pipeline.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"TicketTypeClassifier model not found at {path}")
        self.pipeline = joblib.load(path)
        self.is_trained = True


class ChurnPredictor:
    """
    Simple regression model for churn risk in [0, 100].

    It uses only structured features (age, incidents, sentiment, phishing, word_count)
    so that the dashboard can provide them easily.
    """

    FEATURE_ORDER = [
        "project_age_days",
        "open_incidents_30d",
        "sentiment_label",
        "is_phishing",
        "word_count",
    ]

    def __init__(self, model: Optional[RandomForestRegressor] = None):
        self.model: Optional[RandomForestRegressor] = model
        self.is_trained: bool = model is not None

    def _build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ChurnPredictor":
        if X.empty:
            raise ValueError("No training data provided to ChurnPredictor.fit().")
        self.model = self._build_model()
        self.model.fit(X[self.FEATURE_ORDER], y)
        self.is_trained = True
        return self

    def _to_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict into numpy array with fixed column order."""
        return np.array(
            [[float(features.get(name, 0.0)) for name in self.FEATURE_ORDER]],
            dtype=float,
        )

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict churn risk for a single ticket.

        Args:
            features: dict with keys:
                project_age_days, open_incidents_30d,
                sentiment_label, is_phishing, word_count

        Returns:
            float: churn risk between 0 and 100.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("ChurnPredictor is not trained/loaded.")

        X_vec = self._to_feature_vector(features)
        pred = self.model.predict(X_vec)[0]
        # Clamp to [0, 100] just in case
        return float(max(0.0, min(100.0, pred)))

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized churn model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"ChurnPredictor model not found at {path}")
        self.model = joblib.load(path)
        self.is_trained = True


# -------------------
# High-level trainer
# -------------------


@dataclass
class ModelTrainer:
    """
    Helper class that encapsulates the full training pipeline.

    Usage:
        trainer = create_model_trainer()
        df = trainer.load_data()
        trainer.train_ticket_classifier(df)
        trainer.train_churn_predictor(df)
    """

    data_path: Path
    models_dir: Path

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found at {self.data_path.resolve()}"
            )

        df = pd.read_csv(self.data_path)

        # Basic sanity checks
        expected_cols = {
            "ticket_id",
            "client_name",
            "project_name",
            "channel",
            "text",
            "ticket_type",
            "churn_risk",
            "project_age_days",
            "open_incidents_30d",
            "sentiment_label",
            "is_phishing",
            "has_pii",
        }
        missing = expected_cols.difference(df.columns)
        if missing:
            # No abort; just warn – hackathon friendly
            print(f"⚠️  Dataset missing columns: {sorted(missing)}")

        # Derive word_count for churn model
        df["word_count"] = df["text"].fillna("").str.split().str.len()

        return df

    # --- Training routines ---

    def train_ticket_classifier(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[TicketTypeClassifier, Dict[str, float]]:
        if df is None:
            df = self.load_data()

        X = df["text"].astype(str)
        y = df["ticket_type"].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = TicketTypeClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print("\n📊 Ticket Type Classifier metrics")
        print("--------------------------------")
        print(f"Accuracy: {acc:.3f}")
        print(f"F1-score (weighted): {f1:.3f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        # Persist model
        model_path = self.models_dir / "ticket_classifier.pkl"
        clf.save(model_path)
        print(f"💾 Saved ticket classifier to {model_path.resolve()}")

        metrics = {"accuracy": acc, "f1_weighted": f1}
        return clf, metrics

    def train_churn_predictor(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[ChurnPredictor, Dict[str, float]]:
        if df is None:
            df = self.load_data()

        feature_cols = [
            "project_age_days",
            "open_incidents_30d",
            "sentiment_label",
            "is_phishing",
            "word_count",
        ]
        X = df[feature_cols].copy()
        y = df["churn_risk"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg = ChurnPredictor()
        reg.fit(X_train, y_train)

        y_pred = reg.model.predict(X_test[reg.FEATURE_ORDER])
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n📊 Churn Predictor metrics")
        print("-------------------------")
        print(f"MAE: {mae:.3f}")
        print(f"R²:  {r2:.3f}")

        # Persist model
        model_path = self.models_dir / "churn_predictor.pkl"
        reg.save(model_path)
        print(f"💾 Saved churn predictor to {model_path.resolve()}")

        metrics = {"mae": mae, "r2": r2}
        return reg, metrics


# Public factory

def create_model_trainer() -> ModelTrainer:
    """
    Create a ModelTrainer with project-relative default paths.
    """
    data_path = Path("data") / "tickets_train.csv"
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    return ModelTrainer(data_path=data_path, models_dir=models_dir)
