"""
Módulo de preprocesamiento de texto:
- Limpieza
- Sentimiento (lexicón en español)
- Conteo de palabras

Owner: NLP Team
"""

import re
from typing import Dict, List


# -------------------------------------------------------------------
# LEXICÓN DE SENTIMIENTO EN ESPAÑOL (expresiones coloquiales incluidas)
# -------------------------------------------------------------------
SENTIMENT_LEXICON: Dict[str, List[str]] = {
    "positive": [
        "bacano",
        "chévere",
        "brutal",
        "genial",
        "excelente",
        "buenísimo",
        "una chimba",
        "útil",
        "rápido",
        "eficiente",
        "cumplidos",
        "qué nota",
        "súper bien",
        "muchas gracias",
        "agradecido",
        "agradecida",
        "todo perfecto",
        "quedó perfecto",
        "funciona bien",
        "quedó una belleza",
        "impecable",
        "tremendo trabajo",
        "muy atentos",
        "muy amables",
        "buena vibra",
        "buena energía",
        "colaboración",
        "solucionado",
        "resuelto",
        "me ayudaron rápido",
        "buena respuesta",
        "quedó claro",
        "quedó mejor",
        "confiable",
        "buena atención",
        "buena gestión",
        "excelente soporte",
    ],
    "negative": [
        "maluco",
        "horrible",
        "pésimo",
        "lento",
        "muy demorado",
        "no sirve",
        "no funciona",
        "volvió a fallar",
        "queda colgado",
        "no carga",
        "vuelve y juega",
        "me tiene mamado",
        "me tiene cansado",
        "estoy varado",
        "estoy colgado",
        "enredado",
        "esto es un complique",
        "quedó mal",
        "quedó peor",
        "quedó incompleto",
        "no me dieron respuesta",
        "nadie responde",
        "estoy esperando",
        "estoy sin solución",
        "es un lío",
        "esto es un desorden",
        "muy mala atención",
        "muy mala gestión",
        "no cumplen",
        "me dejaron botado",
        "pésimo soporte",
        "muy mala experiencia",
        "frustrante",
        "desesperante",
        "quedé sin acceso",
        "se cayó el sistema",
        "se dañó otra vez",
        "urgentemente",
        "ya van varias veces",
        "sigo igual",
        "nada que ver",
    ],
    "churn_risk": [
        "estoy cansado de esto",
        "estoy aburrido",
        "no aguanto más",
        "llevo semanas con esto",
        "nadie me ayuda",
        "ya he escrito varias veces",
        "necesito solución hoy",
        "esto afecta mi trabajo",
        "estoy perdiendo plata",
        "me afecta la operación",
        "clientes molestos",
        "me toca buscar otra solución",
        "no puedo seguir así",
        "está afectando al cliente",
        "me va a tocar cambiar de proveedor",
    ],
    "aggressive": [
        "grosería",
        "trato feo",
        "mala actitud",
        "contestación fea",
        "regañar",
        "tono feo",
        "maltrato",
        "falta de respeto",
        "actitud horrible",
        "atención terrible",
    ],
}

NEGATION_PATTERNS = [
    r"no\s+han\s+solucionado",
    r"no\s+se\s+ha\s+solucionado",
    r"no\s+está\s+solucionado",
    r"no\s+han\s+arreglado",
    r"no\s+se\s+ha\s+arreglado",
    r"no\s+funciona",
    r"no\s+sirve",
    r"no\s+resuelve\s+nada",
]


class LexiconSentimentAnalyzer:
    """
    Analizador de sentimiento basado en lexicón en español.

    Usa cuatro listas:
    - positive: expresiones positivas
    - negative: expresiones negativas
    - churn_risk: frases que indican riesgo de churn
    - aggressive: frases de tono agresivo / mala atención

    Resultado: score entre -1 (muy negativo) y 1 (muy positivo).
    """

    def __init__(self) -> None:
        # Normalizamos todo a minúsculas y lo guardamos como conjuntos
        self.positive_words = {w.lower() for w in SENTIMENT_LEXICON["positive"]}
        self.negative_words = {w.lower() for w in SENTIMENT_LEXICON["negative"]}
        self.churn_words = {w.lower() for w in SENTIMENT_LEXICON["churn_risk"]}
        self.aggressive_words = {w.lower() for w in SENTIMENT_LEXICON["aggressive"]}

    def analyze(self, text: str) -> float:
        """
        Devuelve un score de -1 (muy negativo) a 1 (muy positivo).

        - Si no se encuentra ninguna palabra / frase, score = 0 (neutral).
        - churn_risk y aggressive se suman al lado negativo.
        - Maneja negaciones simples: "no ... solucionado", "no funciona", etc.
        """
        if not text:
            return 0.0

        txt = text.lower()

        pos_count = 0
        neg_count = 0

        # 1) Positivas, pero revisando si están negadas cerca
        for w in self.positive_words:
            if w in txt:
                pattern = r"(no|nunca|jamás|ningún|ninguna|ninguno)\s+(?:\w+\s+){0,3}" + re.escape(w)
                if re.search(pattern, txt):
                    # Ej: "no quedó solucionado" -> contar como negativo
                    neg_count += 1
                else:
                    pos_count += 1

        # 2) Negativas + churn + agresivo
        neg_count += sum(1 for w in self.negative_words if w in txt)
        neg_count += sum(1 for w in self.churn_words if w in txt)
        neg_count += sum(1 for w in self.aggressive_words if w in txt)

        # 3) Penalización extra para frases típicas de queja
        if any(re.search(p, txt) for p in NEGATION_PATTERNS):
            neg_count += 1
            # Evitar que "solucionado"/"resuelto" se queden como positivos en estas frases
            for w in ["solucionado", "resuelto"]:
                if w in txt and pos_count > 0:
                    pos_count -= 1

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        score = (pos_count - neg_count) / total
        return round(score, 3)

# ---------------------------------------------
# FUNCIÓN GLOBAL DE SENTIMIENTO (COMPATIBILIDAD)
# ---------------------------------------------

# Instancia global para reutilizar y no recrear el analizador cada vez
_global_sentiment_analyzer = LexiconSentimentAnalyzer()


def sentiment_score(text: str) -> float:
    """
    Función de compatibilidad usada por otros módulos
    (por ejemplo dashboard_app.py).

    Retorna un puntaje entre -1 (muy negativo) y 1 (muy positivo),
    usando el lexicón en español.
    """
    return _global_sentiment_analyzer.analyze(text)


class TextPreprocessor:
    """
    Preprocesa texto para análisis:
    - Limpieza
    - Sentimiento (lexicón español)
    - Conteo de palabras
    """

    def __init__(self) -> None:
        self.sentiment_analyzer = LexiconSentimentAnalyzer()

    # -------------------
    # LIMPIEZA DE TEXTO
    # -------------------
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto:
        - Convierte a minúsculas
        - Elimina caracteres especiales
        - Normaliza espacios múltiples

        Mantiene letras con acentos y la ñ.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        # Permitimos letras (incluyendo acentos), números y espacios
        text = re.sub(r"[^a-z0-9áéíóúñ\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -------------------
    # PIPELINE COMPLETO
    # -------------------
    def process(self, text: str) -> dict:
        """
        Procesa el texto completo y retorna un diccionario con:
        - cleaned_text
        - sentiment_score (-1 a 1)
        - word_count
        """
        cleaned = self.clean_text(text)
        score = self.sentiment_analyzer.analyze(cleaned)
        word_count = len(cleaned.split()) if cleaned else 0

        return {
            "cleaned_text": cleaned,
            "sentiment_score": score,
            "word_count": word_count,
        }


# -------------------------
# FACTORY PRINCIPAL
# -------------------------
def create_text_preprocessor() -> TextPreprocessor:
    """Factory para crear el preprocesador."""
    return TextPreprocessor()


# ------------------------------------------------------
# COMPATIBILIDAD LEGACY PARA security.py y otros módulos
# ------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Compatibilidad: función global usada por otros módulos (p.ej. security.py).
    Internamente usa el preprocesador moderno.
    """
    return TextPreprocessor().clean_text(text)
