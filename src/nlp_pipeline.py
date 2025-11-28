"""
Pipeline de NLP para procesamiento completo de tickets.

Orquesta todos los componentes:
- Seguridad (phishing, PII)
- Preprocesamiento (limpieza, sentimiento)
- Predicción (tipo, churn)
- Recomendaciones

Owner: Data Engineering Team
"""

from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from security import create_security_analyzer
from preprocessing import create_text_preprocessor
from models import TicketTypeClassifier, ChurnPredictor
from recommender import create_ticket_recommender


@dataclass
class TicketAnalysisResult:
    """
    Resultado del análisis completo de un ticket.
    
    Contiene toda la información procesada y predicciones.
    """
    # Datos originales
    original_text: str
    
    # Seguridad
    is_phishing: bool
    has_pii: bool
    
    # Procesamiento
    cleaned_text: str
    sentiment_score: float
    word_count: int
    
    # Predicciones
    ticket_type_pred: str
    churn_risk_pred: float
    risk_segment: str
    
    # Recomendaciones
    recommendation_text: str
    
    def to_dict(self) -> Dict:
        """Convierte el resultado a diccionario."""
        return {
            'original_text': self.original_text,
            'is_phishing': self.is_phishing,
            'has_pii': self.has_pii,
            'cleaned_text': self.cleaned_text,
            'sentiment_score': self.sentiment_score,
            'word_count': self.word_count,
            'ticket_type_pred': self.ticket_type_pred,
            'churn_risk_pred': self.churn_risk_pred,
            'risk_segment': self.risk_segment,
            'recommendation_text': self.recommendation_text
        }


class TicketProcessingPipeline:
    """
    Pipeline completo para procesar tickets.
    
    Implementa el patrón Facade simplificando la interacción
    con múltiples subsistemas complejos.
    """
    
    def __init__(
        self,
        ticket_classifier: TicketTypeClassifier,
        churn_predictor: ChurnPredictor,
        models_loaded: bool = False
    ):
        """
        Inicializa el pipeline con todos los componentes.
        
        Args:
            ticket_classifier: Clasificador de tipo de ticket
            churn_predictor: Predictor de churn
            models_loaded: Si los modelos ya están cargados
        """
        # Componentes independientes (factories)
        self.security_analyzer = create_security_analyzer()
        self.text_preprocessor = create_text_preprocessor()
        self.recommender = create_ticket_recommender()
        
        # Modelos de ML (inyectados)
        self.ticket_classifier = ticket_classifier
        self.churn_predictor = churn_predictor
        self.models_loaded = models_loaded
    
    def process(
        self,
        text: str,
        project_age_days: int = 180,
        open_incidents_30d: int = 2
    ) -> TicketAnalysisResult:
        """
        Procesa un ticket completo a través del pipeline.
        
        Args:
            text: Texto original del ticket
            project_age_days: Antigüedad del proyecto en días
            open_incidents_30d: Incidentes abiertos en últimos 30 días
            
        Returns:
            TicketAnalysisResult: Resultado completo del análisis
        """
        if not self.models_loaded:
            raise ValueError(
                "Los modelos no están cargados. "
                "Usa load_models() o entrena los modelos primero."
            )
        
        # 1. Análisis de seguridad
        is_phishing, has_pii = self.security_analyzer.analyze(text)
        
        # 2. Preprocesamiento
        preprocessed = self.text_preprocessor.process(text)
        cleaned_text = preprocessed['cleaned_text']
        sentiment_score = preprocessed['sentiment_score']
        word_count = preprocessed['word_count']
        
        # 3. Predicción de tipo de ticket
        ticket_type_pred = self.ticket_classifier.predict(cleaned_text)
        
        # 4. Predicción de churn
        churn_features = {
            'project_age_days': project_age_days,
            'open_incidents_30d': open_incidents_30d,
            'sentiment_label': sentiment_score,  # Nota: coincide con nombre en training
            'is_phishing': int(is_phishing),
            'word_count': word_count
        }
        churn_risk_pred = self.churn_predictor.predict(churn_features)
        
        # 5. Clasificar segmento de riesgo
        risk_segment = self.recommender.classify_risk_segment(churn_risk_pred)
        
        # 6. Generar recomendaciones
        recommendation_data = {
            'churn_risk': churn_risk_pred,
            'ticket_type': ticket_type_pred,
            'risk_segment': risk_segment,
            'is_phishing': is_phishing,
            'has_pii': has_pii
        }
        recommendation_text = self.recommender.generate_recommendation(recommendation_data)
        
        # Crear resultado
        return TicketAnalysisResult(
            original_text=text,
            is_phishing=is_phishing,
            has_pii=has_pii,
            cleaned_text=cleaned_text,
            sentiment_score=sentiment_score,
            word_count=word_count,
            ticket_type_pred=ticket_type_pred,
            churn_risk_pred=churn_risk_pred,
            risk_segment=risk_segment,
            recommendation_text=recommendation_text
        )
    
    def load_models(self, models_dir: Path) -> None:
        """
        Carga los modelos entrenados.
        
        Args:
            models_dir: Directorio con los modelos guardados
        """
        classifier_path = models_dir / "ticket_classifier.pkl"
        predictor_path = models_dir / "churn_predictor.pkl"
        
        self.ticket_classifier.load(classifier_path)
        self.churn_predictor.load(predictor_path)
        self.models_loaded = True
        
        print("✅ Pipeline listo para procesar tickets")


# Factory para crear pipeline
def create_pipeline(models_dir: Optional[Path] = None) -> TicketProcessingPipeline:
    """
    Crea y configura un pipeline de procesamiento de tickets.
    
    Args:
        models_dir: Directorio con modelos entrenados (carga automáticamente)
        
    Returns:
        TicketProcessingPipeline: Pipeline configurado
    """
    # Crear componentes de ML
    classifier = TicketTypeClassifier()
    predictor = ChurnPredictor()
    
    # Crear pipeline
    pipeline = TicketProcessingPipeline(
        ticket_classifier=classifier,
        churn_predictor=predictor,
        models_loaded=False
    )
    
    # Si se proporciona directorio de modelos, cargarlos
    if models_dir and models_dir.exists():
        pipeline.load_models(models_dir)
    
    return pipeline
