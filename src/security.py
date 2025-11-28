import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessing import clean_text  # import local (dentro de src)

analyzer = SentimentIntensityAnalyzer()


REGEX_EMAIL = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
REGEX_PHONE = r"\b3\d{2}\d{7}\b"         
REGEX_DNI = r"\b[1-9]\d{6,8}\b"         
REGEX_IP = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
REGEX_TOKEN = r"(?i)(token|bearer|authorization)[:=]\s*[\w\-\.]+"
REGEX_URL = r"http[s]?://\S+|bit\.ly|tinyurl|\.ru|\.xyz|\.tk"

AGGRESSIVE_WORDS = [
    "mierda", "estupido", "inutil", "asco", "maldito", "porqueria",
    "imbecil", "hpt", "hp", "terrible", "cansado", "frustrado",
    "inaceptable", "horrible", "odioso", "odie"
]

PHISHING_STRONG = [
    "su cuenta ha sido suspendida",
    "verifique su identidad",
    "alerta de seguridad",
    "actividad inusual",
    "bloqueo",
    "hackearon",
    "intruso detectado",
    "restablecer su contraseña",
    "fraude",
    "cuenta bloqueada"
]

PHISHING_MEDIUM = [
    "correo sospechoso",
    "sospechoso",
    "problema de seguridad",
    "riesgo",
    "advertencia",
    "comprometido",
    "actividad sospechosa"
]



def detect_pii(text: str) -> dict:
    """
    Devuelve diccionario con listas de coincidencias de PII y leaks.
    """
    if not isinstance(text, str):
        return {"email": [], "phone": [], "dni": [], "ip": [], "token": []}

    return {
        "email": re.findall(REGEX_EMAIL, text),
        "phone": re.findall(REGEX_PHONE, text),
        "dni": re.findall(REGEX_DNI, text),
        "ip": re.findall(REGEX_IP, text),
        "token": re.findall(REGEX_TOKEN, text),
    }


def pii_mask(text: str) -> str:
    """
    Reemplaza PII por [MASKED_...], cuidando espacios para que no quede pegado.
    """
    if not isinstance(text, str):
        return text

    masked = text
  
    masked = re.sub(REGEX_EMAIL, " [MASKED_EMAIL] ", masked)
    masked = re.sub(REGEX_PHONE, " [MASKED_PHONE] ", masked)
    masked = re.sub(REGEX_DNI, " [MASKED_DNI] ", masked)
    masked = re.sub(REGEX_IP, " [MASKED_IP] ", masked)
    masked = re.sub(REGEX_TOKEN, " [MASKED_TOKEN] ", masked)

    # Normalizar espacios (evitar dobles espacios)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked



def detect_phishing(text: str) -> dict:
    """
    Scoring simple por reglas:
    - strong words: +30
    - medium words: +12
    - suspicious URL: +40
    Resultado: score 0-100, level BAJO/MEDIO/ALTO, is_phishing boolean.
    """
    if not isinstance(text, str):
        return {"score": 0, "found": [], "level": "BAJO", "is_phishing": False}

    t = text.lower()
    found = []
    score = 0


    for w in PHISHING_STRONG:
        if w in t:
            found.append(w)
            score += 30


    for w in PHISHING_MEDIUM:
        if w in t:
            found.append(w)
            score += 12


    if re.search(REGEX_URL, t):
        found.append("url_sospechosa")
        score += 40

    score = min(score, 100)
    level = "ALTO" if score >= 70 else ("MEDIO" if score >= 30 else "BAJO")
    return {"score": score, "found": found, "level": level, "is_phishing": score >= 30}



def sentiment_score(text: str) -> dict:
    """
    Devuelve {'score': float, 'label': str}
    - Usa VADER
    - Si hay señales de phishing, forzamos sesgo negativo leve
    """
    if not isinstance(text, str):
        return {"score": 0.0, "label": "Neutral"}

    clean = clean_text(text)
    base = analyzer.polarity_scores(clean)["compound"]

 
    phishing = detect_phishing(text)
    if phishing["score"] >= 30:
        base = min(base, -0.25)


    if base <= -0.5:
        label = "Muy Negativo"
    elif base <= -0.1:
        label = "Negativo"
    elif base < 0.1:
        label = "Neutral"
    elif base < 0.5:
        label = "Positivo"
    else:
        label = "Muy Positivo"

    return {"score": round(base, 4), "label": label}



def aggressiveness_score(text: str) -> dict:
    if not isinstance(text, str):
        return {"score": 0, "found": [], "is_aggressive": False}
    t = text.lower()
    hits = [w for w in AGGRESSIVE_WORDS if w in t]
    score = min(len(hits) * 25, 100)
    return {"score": score, "found": hits, "is_aggressive": score >= 50}



def full_security_pipeline(text: str) -> dict:
    """
    Ejecuta todo y devuelve un objeto compuesto.
    """
    if not isinstance(text, str):
        return {}

    pii = detect_pii(text)
    masked = pii_mask(text)
    phishing = detect_phishing(text)
    sentiment = sentiment_score(text)  
    aggression = aggressiveness_score(text)


    if phishing["is_phishing"] and sentiment["score"] > -0.1:
        sentiment["score"] = min(sentiment["score"], -0.25)
        sentiment["label"] = "Negativo"

    return {
        "pii": pii,
        "masked": masked,
        "phishing": phishing,
        "sentiment": sentiment,
        "aggressiveness": aggression
    }
