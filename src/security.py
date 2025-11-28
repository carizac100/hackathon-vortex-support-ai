import re
from typing import Tuple
from abc import ABC, abstractmethod


class SecurityDetector(ABC):
    """
    Clase base abstracta para detectores de seguridad.
    
    Implementa el principio Open/Closed: abierto para extensión,
    cerrado para modificación.
    """
    
    @abstractmethod
    def detect(self, text: str) -> bool:
        """
        Detecta si el texto contiene elementos de seguridad específicos.
        
        Args:
            text: Texto del ticket a analizar
            
        Returns:
            bool: True si se detecta la amenaza/riesgo, False en caso contrario
        """
        pass


class PhishingDetector(SecurityDetector):
    """
    Detector de intentos de phishing en tickets.
    
    Busca patrones comunes de phishing como:
    - Enlaces sospechosos con palabras clave
    - Urgencia para hacer clic
    - Solicitudes de verificación de cuenta
    """
    
    def _init_(self):
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
        """
        Detecta si el texto contiene patrones de phishing.
        
        Args:
            text: Texto del ticket
            
        Returns:
            bool: True si detecta phishing, False en caso contrario
        """
        text_lower = text.lower()
        
        for pattern in self.phishing_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False


class PIIDetector(SecurityDetector):
    """
    Detector de información personal identificable (PII).
    
    Detecta:
    - Direcciones de email
    - Números de teléfono colombianos (formato 3XX XXX XXXX)
    - Números de identificación (cédula colombiana)
    """
    
    def _init_(self):
        """Inicializa el detector con expresiones regulares para PII."""
        # Patrón para emails
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Patrón para teléfonos colombianos (ej: 3112345678, 311-234-5678, 311 234 5678)
        self.phone_pattern = r'\b3\d{2}[-\s]?\d{3}[-\s]?\d{4}\b'
        
        # Patrón para cédula colombiana (8-10 dígitos)
        self.cc_pattern = r'\b\d{8,10}\b'
    
    def detect(self, text: str) -> bool:
        """
        Detecta si el texto contiene información personal.
        
        Args:
            text: Texto del ticket
            
        Returns:
            bool: True si detecta PII, False en caso contrario
        """
        # Buscar emails
        if re.search(self.email_pattern, text):
            return True
        
        # Buscar teléfonos
        if re.search(self.phone_pattern, text):
            return True
        
        # Para cédulas, ser más conservador (evitar falsos positivos)
        # Solo marcar si hay contexto de identificación
        if re.search(r'(c\.?c\.?|cédula|cedula|id|identification)', text.lower()):
            if re.search(self.cc_pattern, text):
                return True
        
        return False


class SecurityAnalyzer:
    """
    Orquestador de análisis de seguridad.
    
    Implementa el principio de Inversión de Dependencias (DIP):
    depende de abstracciones (SecurityDetector) no de implementaciones concretas.
    """
    
    def __init__(self, phishing_detector: SecurityDetector, pii_detector: SecurityDetector):
        """
        Inicializa el analizador con detectores inyectados.
        
        Args:
            phishing_detector: Detector de phishing
            pii_detector: Detector de PII
        """
        self.phishing_detector = phishing_detector
        self.pii_detector = pii_detector
    
    def analyze(self, text: str) -> Tuple[bool, bool]:
        """
        Analiza el texto para detectar amenazas de seguridad.
        
        Args:
            text: Texto del ticket a analizar
            
        Returns:
            Tuple[bool, bool]: (is_phishing, has_pii)
        """
        is_phishing = self.phishing_detector.detect(text)
        has_pii = self.pii_detector.detect(text)
        
        return is_phishing, has_pii


# Factory para crear instancia por defecto (patrón Factory)
def create_security_analyzer() -> SecurityAnalyzer:
    """
    Crea una instancia de SecurityAnalyzer con detectores por defecto.
    
    Returns:
        SecurityAnalyzer: Analizador configurado y listo para usar
    """
    phishing_detector = PhishingDetector()
    pii_detector = PIIDetector()
    
    return SecurityAnalyzer(phishing_detector, pii_detector)