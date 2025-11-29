"""
Módulo de preprocesamiento de texto:
- Limpieza
- Sentimiento
- Conteo de palabras

Owner: NLP Team
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> float:
    """Retorna el puntaje de sentimiento entre -1 y 1."""
    return analyzer.polarity_scores(text)["compound"]


class TextPreprocessor:
    """
    Preprocesa texto para análisis:
    - Limpieza
    - Sentimiento
    - Conteo
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def clean_text(self, text: str) -> str:
        """Limpia el texto eliminando caracteres especiales y múltiples espacios."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9áéíóúñ\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process(self, text: str):
        """Procesa el texto completo y retorna un diccionario."""
        cleaned = self.clean_text(text)

        sentiment_score = self.analyzer.polarity_scores(cleaned)["compound"]
        word_count = len(cleaned.split()) if cleaned else 0

        return {
            "cleaned_text": cleaned,
            "sentiment_score": sentiment_score,
            "word_count": word_count
        }


def create_text_preprocessor() -> TextPreprocessor:
    """Factory para crear el preprocesador."""
    return TextPreprocessor()

# --- COMPATIBILIDAD LEGACY PARA security.py ---
# Security.py necesita esta función para no romper el código

def clean_text(text: str) -> str:
    """
    Compatibilidad: función global usada por security.py
    Internamente usa el preprocesador moderno.
    """
    return TextPreprocessor().clean_text(text)