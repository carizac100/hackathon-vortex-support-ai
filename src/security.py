import re
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod


from preprocessing import clean_text

# -------------------------------------------
# EXPRESIONES REGULARES PARA PII
# -------------------------------------------

REGEX_EMAIL = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
# Teléfonos colombianos tipo 3XXXXXXXXX (10 dígitos comenzando por 3)
REGEX_PHONE = r"\b3\d{9}\b"
# Cédulas / IDs genéricos (7 a 10 dígitos)
REGEX_DNI = r"\b[1-9]\d{6,9}\b"
# Direcciones IP
REGEX_IP = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
# Tokens / credenciales
REGEX_TOKEN = r"(?i)(token|bearer|authorization)[:=]\s*[\w\-\.]+"
# URLs sospechosas (acortadores y dominios raros)
REGEX_URL = r"http[s]?://\S+|bit\.ly|tinyurl|goo\.gl|\.ru\b|\.xyz\b|\.tk\b"


# --- LÉXICO AGRESIVO (COHERENTE CON preprocessing.py) ---
_AGGRESSIVE_LEXICON = [
    r"\b(grosería|trato\s+feo|mala\s+actitud|contestación\s+fea|regañar|tono\s+feo|maltrato|falta\s+de\s+respeto|actitud\s+horrible|atención\s+terrible)\b"
]

_AGGRESSIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _AGGRESSIVE_LEXICON]


# -------------------------------------------
# CLASE BASE DE DETECTORES
# -------------------------------------------

class SecurityDetector(ABC):
    """
    Clase base abstracta para detectores de seguridad.
    """
    @abstractmethod
    def detect(self, text: str) -> bool:
        """Detecta si el texto contiene elementos de seguridad específicos."""
        raise NotImplementedError


# -------------------------------------------
# DETECTOR DE PII
# -------------------------------------------

class PiiDetector(SecurityDetector):
    """Detector de información personal identificable (PII)."""

    def __init__(self) -> None:
        self._patterns = {
            "emails": re.compile(REGEX_EMAIL),
            "phones": re.compile(REGEX_PHONE),
            "dnis": re.compile(REGEX_DNI),
            "ips": re.compile(REGEX_IP),
            "tokens": re.compile(REGEX_TOKEN),
        }

    def detect(self, text: str) -> bool:
        """Devuelve True si encuentra cualquier tipo de PII."""
        matches = self.find_all(text)
        return any(matches[key] for key in matches)

    def find_all(self, text: str) -> Dict[str, List[str]]:
        """Devuelve todas las coincidencias de PII por tipo."""
        if text is None:
            return {k: [] for k in self._patterns.keys()}

        result: Dict[str, List[str]] = {}
        for key, pattern in self._patterns.items():
            result[key] = pattern.findall(text)
        return result

    def mask(self, text: str) -> str:
        """Enmascara PII en el texto con tags genéricos."""
        if text is None:
            return ""

        masked = text
        masked = re.sub(REGEX_EMAIL, "[EMAIL]", masked)
        masked = re.sub(REGEX_PHONE, "[PHONE]", masked)
        masked = re.sub(REGEX_DNI, "[ID]", masked)
        masked = re.sub(REGEX_IP, "[IP]", masked)
        masked = re.sub(REGEX_TOKEN, "[TOKEN]", masked)
        return masked


# -------------------------------------------
# DETECTOR DE PHISHING
# -------------------------------------------

class PhishingDetector(SecurityDetector):
    """Detector simple de patrones de phishing en el texto."""

    def __init__(self) -> None:
        # Frases típicas de phishing en español
        self._patterns = [
            r"actualiza\s+tu\s+contraseña",
            r"actualice\s+su\s+contraseña",
            r"verifica[r]?\s+tu\s+cuenta",
            r"verifique\s+su\s+cuenta",
            r"tu\s+cuenta\s+será\s+cerrada",
            r"ha\s+sido\s+bloqueada",
            r"haz\s+clic\s+en\s+el\s+enlace",
            r"haga\s+clic\s+en\s+el\s+enlace",
            r"ingrese\s+sus\s+datos",
            r"proporcione\s+sus\s+datos",
            r"confirme\s+su\s+información",
            r"urgente",
            r"urgentemente",
            r"debe\s+actuar\s+de\s+inmediato",
        ]
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self._patterns]

    def detect(self, text: str) -> bool:
        if not text:
            return False
        txt = text.lower()
        return any(p.search(txt) for p in self._compiled)


# -------------------------------------------
# DETECTOR DE LENGUAJE AGRESIVO
# -------------------------------------------

class AggressiveLanguageDetector(SecurityDetector):
    """Detector de lenguaje agresivo o inapropiado."""

    def detect(self, text: str) -> bool:
        """Detecta si el texto contiene patrones de lenguaje agresivo."""
        if not text:
            return False
        txt = text.lower()
        return any(p.search(txt) for p in _AGGRESSIVE_PATTERNS)


# -------------------------------------------
# ANALIZADOR PRINCIPAL DE SEGURIDAD
# -------------------------------------------

class SecurityAnalyzer:
    """
    Analizador de seguridad que combina:
    - PII
    - Phishing
    - Lenguaje agresivo
    """

    def __init__(
        self,
        phishing_detector: SecurityDetector,
        pii_detector: PiiDetector,
        aggressive_detector: SecurityDetector,
    ) -> None:
        self.phishing_detector = phishing_detector
        self.pii_detector = pii_detector
        self.aggressive_detector = aggressive_detector

    def analyze(self, text: str) -> Tuple[bool, bool]:
        """
        Usado por el pipeline de NLP.
        Devuelve:
            (is_phishing, has_pii)
        """
        normalized = clean_text(text)
        is_phishing = self.phishing_detector.detect(normalized)
        has_pii = self.pii_detector.detect(normalized)
        return is_phishing, has_pii

    # --- Métodos usados desde el dashboard (funciones globales) ---

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        normalized = text or ""
        return self.pii_detector.find_all(normalized)

    def mask_pii(self, text: str) -> str:
        normalized = text or ""
        return self.pii_detector.mask(normalized)

    def detect_phishing(self, text: str) -> bool:
        normalized = clean_text(text)
        return self.phishing_detector.detect(normalized)

    def detect_aggressive_language(self, text: str) -> bool:
        normalized = clean_text(text)
        return self.aggressive_detector.detect(normalized)


# -------------------------------------------
# FACTORY Y FUNCIONES GLOBALES
# -------------------------------------------

# Instancia global reutilizable
_analyzer = SecurityAnalyzer(
    phishing_detector=PhishingDetector(),
    pii_detector=PiiDetector(),
    aggressive_detector=AggressiveLanguageDetector(),
)


def create_security_analyzer() -> SecurityAnalyzer:
    """
    Factory usado por nlp_pipeline.py

    En nlp_pipeline se hace:
        self.security_analyzer = create_security_analyzer()
        is_phishing, has_pii = self.security_analyzer.analyze(text)
    """
    return _analyzer


def detect_pii(text: str):
    """Llama al método detect_pii del analizador."""
    return _analyzer.detect_pii(text)


def mask_pii(text: str):
    """Llama al método mask_pii del analizador."""
    return _analyzer.mask_pii(text)


def detect_phishing(text: str):
    """Llama al método detect_phishing del analizador."""
    return _analyzer.detect_phishing(text)


def detect_aggressive_language(text: str):
    """Llama al método detect_aggressive_language del analizador."""
    return _analyzer.detect_aggressive_language(text)
