import re
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

# --- NUEVA FUNCIONALIDAD: LÉXICO AGRESIVO ---
# Palabras clave para detectar lenguaje ofensivo o agresivo del cliente hacia el agente.
_AGGRESSIVE_LEXICON = [
    r'\b(grosería|trato\s+feo|mala\s+actitud|contestación\s+fea|regañar|tono\s+feo|maltrato|falta\s+de\s+respeto|actitud\s+horrible|atención\s+terrible)\b'
]
# Las expresiones regulares se compilan para un mejor rendimiento.
_AGGRESSIVE_PATTERNS = [re.compile(p) for p in _AGGRESSIVE_LEXICON]
# ---------------------------------------------


# --- DETECTORES DE SEGURIDAD (IMPLEMENTACIONES CONCRETAS) ---

class SecurityDetector(ABC):
    """
    Clase base abstracta para detectores de seguridad.
    """
    @abstractmethod
    def detect(self, text: str) -> bool:
        """Detecta si el texto contiene elementos de seguridad específicos."""
        pass

# --- NUEVA FUNCIONALIDAD: DETECTOR DE LENGUAJE AGRESIVO ---
class AggressiveLanguageDetector(SecurityDetector):
    """Detector de lenguaje agresivo o inapropiado."""
    
    def detect(self, text: str) -> bool:
        """Detecta si el texto contiene patrones de lenguaje agresivo."""
        text_lower = text.lower()
        
        # Iterar sobre los patrones compilados
        for pattern in _AGGRESSIVE_PATTERNS:
            if pattern.search(text_lower):
                return True
        return False
# -----------------------------------------------------------


class PhishingDetector(SecurityDetector):
    """Detector de intentos de phishing en tickets."""
    
    def __init__(self):
        """Inicializa el detector con patrones de phishing conocidos."""
        # Patrones comunes de phishing en español e inglés
        self.phishing_patterns = [
            r'click\s+(here|aquí|acá)',
            r'verify\s+your\s+account',
            r'verificar?\s+tu\s+cuenta',
            r'reset\s+your\s+password',
            r'restablecer?\s+contraseña',
            r'urgent(ly)?',
            r'urgente(mente)?',
            r'deactivated?',
            r'desactivad[oa]',
            r'suspend(ed)?',
            r'suspend(ido|ida)',
            r'immediately',
            r'inmediatamente',
        ]
    
    def detect(self, text: str) -> bool:
        """Detecta si el texto contiene patrones de phishing."""
        text_lower = text.lower()
        
        for pattern in self.phishing_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False


class PIIDetector(SecurityDetector):
    """Detector de información personal identificable (PII)."""

    # Definimos el patrón de PII completo fuera de la detección
    # para usarlo también en la función de enmascaramiento.
    # Patrones: Emails, Teléfonos (Col), Cédulas (Col).
    _EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    _PHONE_PATTERN = r'\b3\d{2}[-\s]?\d{3}[-\s]?\d{4}\b'
    _CC_PATTERN = r'\b\d{8,10}\b'
    _PII_PATTERN = '|'.join([_EMAIL_PATTERN, _PHONE_PATTERN]) # Unimos solo email y teléfono para detección simple.

    def __init__(self):
        """Inicializa el detector con expresiones regulares para PII."""
        # No requiere inicialización adicional, los patrones son constantes de clase.
        pass
    
    def detect(self, text: str) -> bool:
        """Detecta si el texto contiene información personal."""
        
        # 1. Buscar Email o Teléfono (patrones directos)
        if re.search(self._PII_PATTERN, text):
            return True
        
        # 2. Buscar Cédula (solo si hay contexto de identificación)
        if re.search(r'(c\.?c\.?|cédula|cedula|id|identification|identificaci[oó]n)', text.lower()):
            if re.search(self._CC_PATTERN, text):
                return True
        
        return False

    def mask(self, text: str) -> str:
        """Enmascara la información PII encontrada en el texto."""
        
        # Enmascara emails y teléfonos
        text = re.sub(self._EMAIL_PATTERN, '[EMAIL_MASKED]', text)
        text = re.sub(self._PHONE_PATTERN, '[PHONE_MASKED]', text)

        text = re.sub(self._CC_PATTERN, '[ID_MASKED]', text)
        
        return text


# --- ORQUESTADOR ---
class SecurityAnalyzer:
    """Orquestador de análisis de seguridad."""
    
    def __init__(self, phishing_detector: SecurityDetector, pii_detector: PIIDetector, aggressive_detector: SecurityDetector):
        """
        Inicializa el analizador con detectores inyectados.
        """
        self.phishing_detector = phishing_detector
        self.pii_detector = pii_detector
        self.aggressive_detector = aggressive_detector 
    
    def analyze(self, text: str) -> Tuple[bool, bool, bool]:
        """Analiza el texto para detectar amenazas de seguridad."""
        is_phishing = self.phishing_detector.detect(text)
        has_pii = self.pii_detector.detect(text)
        is_aggressive = self.aggressive_detector.detect(text)
        
        return is_phishing, has_pii, is_aggressive

    
    def detect_pii(self, text: str) -> bool:
        """Expone la detección de PII."""
        return self.pii_detector.detect(text)

    def detect_phishing(self, text: str) -> bool:
        """Expone la detección de phishing."""
        return self.phishing_detector.detect(text)
    
    # --- EXPOSICIÓN DE DETECCIÓN AGRESIVA ---
    def detect_aggressive_language(self, text: str) -> bool:
        """Expone la detección de lenguaje agresivo."""
        return self.aggressive_detector.detect(text)
    # -------------------------------------------------------------

    def mask_pii(self, text: str) -> str:
        """Expone el enmascaramiento de PII."""

        return self.pii_detector.mask(text)


# --- FACTORY ---
def create_security_analyzer() -> SecurityAnalyzer:
    """Crea una instancia de SecurityAnalyzer con detectores por defecto."""
    phishing_detector = PhishingDetector()
    pii_detector = PIIDetector()
    aggressive_detector = AggressiveLanguageDetector() 
    

    return SecurityAnalyzer(phishing_detector, pii_detector, aggressive_detector) 


_analyzer = create_security_analyzer() 

# --- FUNCIONES DE CONVENIENCIA GLOBALES ---

def detect_pii(text: str):
    """Llama al método detect_pii del analizador."""
    return _analyzer.detect_pii(text)

def mask_pii(text: str):
    """Llama al método mask_pii del analizador."""
    return _analyzer.mask_pii(text)

def detect_phishing(text: str):
    """Llama al método detect_phishing del analizador."""
    return _analyzer.detect_phishing(text)

# --- FUNCIÓN GLOBAL AGRESIVA ---
def detect_aggressive_language(text: str):
    """Llama al método detect_aggressive_language del analizador."""
    return _analyzer.detect_aggressive_language(text)
# ---------------------------------------------------