
import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto para NLP.
    - Minúsculas
    - Quitar caracteres especiales
    - Quitar emojis
    - Quitar espacios dobles
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = unicodedata.normalize("NFKD", text)


    text = re.sub(r"[^a-z0-9áéíóúñ ]", " ", text)


    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str):
    """
    Tokenización simple basada en espacios.
    """
    text = clean_text(text)
    return text.split()
