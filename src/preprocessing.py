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
        "muy cumplidos",
        "funcionó perfecto",
        "muchas gracias",
        "gracias por la ayuda",
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
        "qué nota",
        "súper bien",
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
    ],
    "negative": [
        "maluco",
        "horrible",
        "pésimo",
        "terrible",
        "lento",
        "demorado",
        "no sirve",
        "no funciona",
        "dañado",
        "bloqueado",
        "se cae",
        "se bloquea",
        "traba mucho",
        "traba demasiado",
        "se queda pensando",
        "incómodo",
        "enredado",
        "confuso",
        "no entiendo nada",
        "no me deja entrar",
        "me saca del sistema",
        "muy demorado",
        "muy lento",
        "fatal",
        "una porquería",
        "un asco",
        "me tiene mamado",
        "me tiene cansado",
        "no aguanto más",
        "no han solucionado",
        "sigue igual",
        "peor que antes",
        "nunca funciona",
        "cada rato falla",
        "cada rato se cae",
        "otra vez dañado",
        "sigue fallando",
        "sigue bloqueado",
        "sigue mal",
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
    ],
    "churn_risk": [
        "no aguanto más",
        "me toca buscar otra solución",
        "voy a cancelar",
        "voy a cerrar el contrato",
        "voy a dejar de usar",
        "voy a dejar de trabajar con ustedes",
        "estoy pensando en irme",
        "me va a tocar cambiar de proveedor",
        "estoy cansado del servicio",
        "no puedo seguir así",
        "no me sirve el sistema",
        "ya no confío en la herramienta",
        "estoy perdiendo plata",
        "estoy perdiendo dinero",
        "estoy perdiendo clientes",
        "estos errores me afectan mucho",
        "esto afecta mi negocio",
        "esto afecta mi trabajo",
        "no veo mejora",
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

# Patrones para detectar frases negativas con negación directa
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
    Analizador de sentimiento basado en un lexicón en español.

    Usa cuatro listas:
    - positive: expresiones positivas
    - negative: expresiones negativas
    - churn_risk: frases que indican riesgo de churn
    - aggressive: frases de tono agresivo / mala atención

    Reglas:
    - churn_risk y aggressive se suman al lado negativo.
    - Manejo simple de negaciones: "no funciona", "no han solucionado nada", etc.
    - Devuelve un puntaje entre -1 (muy negativo) y 1 (muy positivo).
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
        - churn_risk y aggressive se cuentan como negativas.
        - Evita marcar como positivo algo que esté negado, por ejemplo:
          "no han solucionado nada", "no quedó resuelto".
        """
        if not text:
            return 0.0

        txt = text.lower()

        pos_count = 0
        neg_count = 0

        # 1) Palabras positivas, verificando si están negadas
        for w in self.positive_words:
            if w in txt:
                # Busca patrones tipo "no ... solucionado"
                pattern = (
                    r"(no|nunca|jamás|ningún|ninguna|ninguno)\s+(?:\w+\s+){0,3}"
                    + re.escape(w)
                )
                if re.search(pattern, txt):
                    # En contexto de negación, contabilizamos como negativo
                    neg_count += 1
                else:
                    pos_count += 1

        # 2) Palabras negativas
        neg_count += sum(1 for w in self.negative_words if w in txt)

        # 3) Frases de churn y agresividad también suman al lado negativo
        neg_count += sum(1 for w in self.churn_words if w in txt)
        neg_count += sum(1 for w in self.aggressive_words if w in txt)

        # 4) Penalización extra por patrones típicos de queja con negación
        if any(re.search(p, txt) for p in NEGATION_PATTERNS):
            neg_count += 1
            # Evitar que "solucionado"/"resuelto" cuenten como positivos aquí
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
    """Función de conveniencia para obtener el puntaje de sentimiento.

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
        - Deja solo letras, números y espacios
        """
        if text is None:
            return ""

        # Normalizamos saltos de línea y espacios
        cleaned = text.replace("\n", " ").replace("\r", " ")
        cleaned = cleaned.strip().lower()

        # Permitimos letras, números, algunos signos básicos y espacios
        cleaned = re.sub(r"[^a-záéíóúñü0-9.,;:!?@#\-\s]", " ", cleaned)

        # Colapsamos múltiples espacios
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # -------------------
    # ANÁLISIS COMPLETO
    # -------------------
    def analyze(self, text: str) -> Dict[str, object]:
        """
        Devuelve un diccionario con:
        - cleaned_text: texto limpio
        - sentiment_score: puntaje de sentimiento (-1 a 1)
        - word_count: conteo de palabras
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
