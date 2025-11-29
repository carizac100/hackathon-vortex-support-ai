"""
NLP pipeline for Neuro Support AI.

Orchestrates:
- Text preprocessing (cleaning + sentiment + word count)
- Security analysis (phishing + PII)
- ML models (ticket type classification + churn regression)
- Rule-based recommendation text.

This module exposes a single public factory:

    create_pipeline(models_dir: Optional[Path]) -> TicketProcessingPipeline

which is used by the Streamlit dashboard.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import joblib

from preprocessing import create_text_preprocessor
from security import create_security_analyzer
from models import TicketTypeClassifier, ChurnPredictor

# ---------------------------------------------------------------------------
# Dataclass used as return type from the pipeline
# ---------------------------------------------------------------------------


@dataclass
class TicketModelResult:
    """
    Container for all model outputs used by the dashboard.

    Attributes
    ----------
    ticket_type_pred: str
        Predicted ticket type label (e.g. "Correctivo", "Evolutivo").
    churn_risk_pred: float
        Churn risk score in [0, 100].
    risk_segment: str
        Segment label derived from churn_risk_pred ("Alto", "Medio", "Bajo").
    recommendation_text: str
        Human-readable recommendation message.
    cleaned_text: str
        Normalized text returned by the preprocessor.
    """

    ticket_type_pred: str
    churn_risk_pred: float
    risk_segment: str
    recommendation_text: str
    cleaned_text: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _sentiment_label_from_score(score: float) -> int:
    """
    Map continuous sentiment score [-1, 1] to discrete label {-1, 0, 1}.

    - score <= -0.25  -> -1 (negative)
    - -0.25 < score < 0.25 -> 0 (neutral)
    - score >= 0.25 -> 1 (positive)
    """
    if score <= -0.25:
        return -1
    if score >= 0.25:
        return 1
    return 0


def _segment_from_churn(churn_risk: float) -> str:
    """
    Map churn risk [0, 100] to human-friendly risk segment.

    - >= 70  -> "Alto"
    - [40,70) -> "Medio"
    - < 40   -> "Bajo"
    """
    if churn_risk >= 70:
        return "Alto"
    if churn_risk >= 40:
        return "Medio"
    return "Bajo"


def _build_recommendation(
    ticket_type: str,
    churn_risk: float,
    sentiment_score: float,
    is_phishing: bool,
    has_pii: bool,
) -> str:
    """
    Create a human readable recommendation text combining the different signals.
    """
    # 1) Seguridad tiene prioridad
    if is_phishing and has_pii:
        return (
            "⚠️ Posible phishing con datos sensibles. "
            "Escala inmediatamente al equipo de seguridad y bloquea cualquier enlace sospechoso."
        )
    if is_phishing:
        return (
            "⚠️ Posible phishing detectado. "
            "Verifica la autenticidad del remitente antes de responder al cliente."
        )
    if has_pii:
        return (
            "📛 Contiene datos sensibles (PII). "
            "Evita copiar esta información a canales no seguros."
        )

    # 2) Riesgo + sentimiento
    if churn_risk >= 70 or sentiment_score <= -0.5:
        return (
            "🔥 Alta probabilidad de churn. Prioriza este ticket, "
            "contacta al cliente por un canal directo y ofrece una solución concreta."
        )
    if churn_risk >= 40 or sentiment_score < 0:
        return (
            "⚠️ Riesgo medio de churn. Responde con empatía, "
            "explica los pasos de solución y haz seguimiento cercano."
        )

    # 3) Riesgo bajo
    if ticket_type.lower().startswith("evolutivo"):
        return (
            "✅ Ticket evolutivo de baja urgencia. "
            "Regístralo en el backlog y comunícale al cliente el tiempo estimado."
        )

    return (
        "✅ Riesgo bajo y sin alertas de seguridad. "
        "Resuelve el ticket siguiendo el flujo estándar de soporte."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class TicketProcessingPipeline:
    """
    High level NLP pipeline used by the Streamlit dashboard.

    It accepts raw ticket text plus a few structured features and returns
    a TicketModelResult with all the derived information.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        # Core components
        self.preprocessor = create_text_preprocessor()
        self.security_analyzer = create_security_analyzer()

        # ML models (loaded lazily from disk)
        self.ticket_classifier: Optional[TicketTypeClassifier] = None
        self.churn_predictor: Optional[ChurnPredictor] = None
        self.models_loaded: bool = False

        if models_dir is not None:
            self.load_models(models_dir)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_models(self, models_dir: Path) -> None:
        """
        Load the trained models from disk using the high level wrappers
        defined in models.py.

        This avoids passing raw text directly into a numeric-only model,
        which is what caused the "could not convert string to float"
        error you were seeing.
        """
        models_dir = Path(models_dir)

        clf_path = models_dir / "ticket_classifier.pkl"
        churn_path = models_dir / "churn_predictor.pkl"

        if not clf_path.exists() or not churn_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {models_dir}. "
                "Run `python src/train_models.py` first."
            )

        # Ticket type classifier
        clf = TicketTypeClassifier()
        clf.load(clf_path)
        self.ticket_classifier = clf

        # Churn predictor (regression)
        churn = ChurnPredictor()
        churn.load(churn_path)
        self.churn_predictor = churn

        self.models_loaded = True

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------
    def process(
        self,
        text: str,
        project_age_days: int,
        open_incidents_30d: int,
    ) -> TicketModelResult:
        """
        Run the full pipeline: preprocessing + security + ML models +
        recommendation generation.
        """
        if not isinstance(text, str):
            text = str(text or "")

        # --- 1. Preprocessing ---
        pre = self.preprocessor.analyze(text)
        cleaned_text: str = pre["cleaned_text"]
        sentiment_score: float = float(pre["sentiment_score"])
        word_count: int = int(pre["word_count"])

        # --- 2. Security analysis ---
        is_phishing, has_pii = self.security_analyzer.analyze(text)

        # --- 3. Ticket type prediction ---
        ticket_label = "No disponible"
        if self.ticket_classifier is not None:
            try:
                # Nuestro wrapper acepta un string o una lista
                ticket_label = str(self.ticket_classifier.predict(cleaned_text))
            except Exception:
                ticket_label = "No disponible"

        # --- 4. Churn prediction ---
        churn_risk = 0.0
        if self.churn_predictor is not None:
            sent_label = _sentiment_label_from_score(sentiment_score)
            is_phishing_int = int(bool(is_phishing))

            features_dict = {
                "project_age_days": float(project_age_days),
                "open_incidents_30d": float(open_incidents_30d),
                "sentiment_label": float(sent_label),
                "is_phishing": float(is_phishing_int),
                "word_count": float(word_count),
            }

            try:
                churn_risk = float(self.churn_predictor.predict(features_dict))
            except Exception:
                churn_risk = 0.0

        # Aseguramos que esté en [0, 100]
        churn_risk = float(np.clip(churn_risk, 0.0, 100.0))

        # --- 5. Segmento de riesgo + recomendación ---
        risk_segment = _segment_from_churn(churn_risk)
        recommendation = _build_recommendation(
            ticket_type=ticket_label,
            churn_risk=churn_risk,
            sentiment_score=sentiment_score,
            is_phishing=is_phishing,
            has_pii=has_pii,
        )

        return TicketModelResult(
            ticket_type_pred=ticket_label,
            churn_risk_pred=churn_risk,
            risk_segment=risk_segment,
            recommendation_text=recommendation,
            cleaned_text=cleaned_text,
        )

# ---------------------------------------------------------------------------
# Public factory used by dashboard_app.py
# ---------------------------------------------------------------------------


def create_pipeline(models_dir: Optional[Path] = None) -> TicketProcessingPipeline:
    """
    Factory called from the Streamlit dashboard.

    Parameters
    ----------
    models_dir : Optional[Path]
        Directory containing the trained model pickle files. If None, the
        pipeline is created without loading models (only preprocessing and
        security will work).
    """
    return TicketProcessingPipeline(models_dir=models_dir)