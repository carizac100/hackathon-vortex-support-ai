"""
Módulo de Preprocesamiento de Texto para tickets de soporte.

Implementa limpieza y normalización de texto para preparar datos
para análisis NLP y modelos de ML.

Owner: Data Science Team
"""

import re
from typing import Protocol
from abc import ABC, abstractmethod


class TextCleaner(ABC):
    """
    Clase base abstracta para limpiadores de texto.
    
    Implementa el principio Open/Closed (OCP).
    """
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Limpia el texto según la estrategia implementada.
        
        Args:
            text: Texto original
            
        Returns:
            str: Texto limpio
        """
        pass


class BasicTextCleaner(TextCleaner):
    """
    Limpiador básico de texto que:
    - Convierte a minúsculas
    - Elimina URLs
    - Elimina emails
    - Elimina números de teléfono
    - Elimina caracteres especiales excesivos
    - Normaliza espacios en blanco
    """
    
    def clean(self, text: str) -> str:
        """
        Aplica limpieza básica al texto.
        
        Args:
            text: Texto original del ticket
            
        Returns:
            str: Texto limpio y normalizado
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convertir a minúsculas
        cleaned = text.lower()
        
        # Eliminar URLs
        cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)
        
        # Eliminar emails (para privacidad)
        cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned)
        
        # Eliminar números de teléfono
        cleaned = re.sub(r'\b3\d{2}[-\s]?\d{3}[-\s]?\d{4}\b', '', cleaned)
        
        # Eliminar cédulas (números largos)
        cleaned = re.sub(r'\b\d{8,10}\b', '', cleaned)
        
        # Mantener solo letras, números, espacios y puntuación básica
        cleaned = re.sub(r'[^a-záéíóúñ0-9\s.,;:!?()-]', ' ', cleaned)
        
        # Normalizar espacios múltiples
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Eliminar espacios al inicio y final
        cleaned = cleaned.strip()
        
        return cleaned


class SentimentAnalyzer:
    """
    Analizador de sentimiento simple basado en palabras clave.
    
    Implementa el principio de Responsabilidad Única (SRP).
    """
    
    def __init__(self):
        """Inicializa el analizador con diccionarios de palabras positivas y negativas."""
        # Palabras negativas relacionadas con problemas de soporte
        self.negative_words = {
            'error', 'fallo', 'problema', 'mal', 'no funciona', 'roto', 'crash',
            'lento', 'frustrado', 'molesto', 'enojado', 'angry', 'frustrated',
            'inaceptable', 'unacceptable', 'terrible', 'pésimo', 'awful',
            'urgente', 'urgent', 'critical', 'crítico', 'bloqueado', 'blocked'
        }
        
        # Palabras positivas o neutras
        self.positive_words = {
            'gracias', 'thanks', 'excelente', 'excellent', 'bien', 'good',
            'perfecto', 'perfect', 'funciona', 'works', 'resuelto', 'solved',
            'agradecer', 'appreciate', 'mejor', 'better', 'mejora', 'improvement'
        }
    
    def analyze(self, text: str) -> float:
        """
        Calcula un score de sentimiento simple para el texto.
        
        Args:
            text: Texto limpio del ticket
            
        Returns:
            float: Score de sentimiento entre -1 (muy negativo) y 1 (muy positivo)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Contar palabras positivas y negativas
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        
        # Calcular score
        total_count = negative_count + positive_count
        
        if total_count == 0:
            return 0.0  # Neutral si no hay palabras de sentimiento
        
        # Score normalizado entre -1 y 1
        score = (positive_count - negative_count) / total_count
        
        return round(score, 3)


class TextPreprocessor:
    """
    Orquestador de preprocesamiento de texto.
    
    Implementa el principio de Inversión de Dependencias (DIP):
    depende de la abstracción TextCleaner.
    """
    
    def __init__(self, cleaner: TextCleaner, sentiment_analyzer: SentimentAnalyzer):
        """
        Inicializa el preprocesador con componentes inyectados.
        
        Args:
            cleaner: Estrategia de limpieza de texto
            sentiment_analyzer: Analizador de sentimiento
        """
        self.cleaner = cleaner
        self.sentiment_analyzer = sentiment_analyzer
    
    def process(self, text: str) -> dict:
        """
        Procesa el texto aplicando limpieza y análisis.
        
        Args:
            text: Texto original del ticket
            
        Returns:
            dict: {
                'cleaned_text': str,
                'sentiment_score': float,
                'word_count': int
            }
        """
        # Limpiar texto
        cleaned = self.cleaner.clean(text)
        
        # Calcular sentimiento
        sentiment = self.sentiment_analyzer.analyze(cleaned)
        
        # Contar palabras
        word_count = len(cleaned.split())
        
        return {
            'cleaned_text': cleaned,
            'sentiment_score': sentiment,
            'word_count': word_count
        }


# Factory para crear instancia por defecto
def create_text_preprocessor() -> TextPreprocessor:
    """
    Crea una instancia de TextPreprocessor con configuración por defecto.
    
    Returns:
        TextPreprocessor: Preprocesador configurado y listo para usar
    """
    cleaner = BasicTextCleaner()
    sentiment_analyzer = SentimentAnalyzer()
    
    return TextPreprocessor(cleaner, sentiment_analyzer)

