"""
Módulo de Recomendaciones para tickets de soporte.

Genera recomendaciones de acción basadas en:
- Tipo de ticket
- Riesgo de churn
- Segmento de riesgo

Owner: Business Intelligence Team
"""

from typing import Dict
from abc import ABC, abstractmethod


class RecommendationStrategy(ABC):
    """
    Estrategia base para generación de recomendaciones.
    
    Implementa el patrón Strategy y el principio Open/Closed (OCP).
    """
    
    @abstractmethod
    def recommend(self, ticket_data: Dict) -> str:
        """
        Genera una recomendación basada en los datos del ticket.
        
        Args:
            ticket_data: Diccionario con información del ticket
            
        Returns:
            str: Texto con la recomendación
        """
        pass


class ChurnBasedRecommendation(RecommendationStrategy):
    """
    Genera recomendaciones basadas en el riesgo de churn.
    """
    
    def recommend(self, ticket_data: Dict) -> str:
        """
        Recomienda acciones según el nivel de riesgo de churn.
        
        Args:
            ticket_data: Debe contener:
                - churn_risk: float (0-100)
                - ticket_type: str ('Correctivo' o 'Evolutivo')
                - risk_segment: str ('Bajo', 'Medio', 'Alto')
                - is_phishing: bool
                - has_pii: bool
                
        Returns:
            str: Recomendación formateada
        """
        churn_risk = ticket_data.get('churn_risk', 0)
        ticket_type = ticket_data.get('ticket_type', 'Desconocido')
        risk_segment = ticket_data.get('risk_segment', 'Medio')
        is_phishing = ticket_data.get('is_phishing', False)
        has_pii = ticket_data.get('has_pii', False)
        
        recommendations = []
        
        # Alertas de seguridad (prioridad máxima)
        if is_phishing:
            recommendations.append("🚨 ALERTA: Posible intento de phishing detectado. Revisar inmediatamente y notificar al equipo de seguridad.")
        
        if has_pii:
            recommendations.append("⚠️ PRIVACIDAD: El ticket contiene información personal. Anonimizar antes de procesar.")
        
        # Recomendaciones por segmento de riesgo
        if risk_segment == 'Alto':
            recommendations.append(
                f"🔴 RIESGO ALTO DE CHURN ({churn_risk:.0f}%): "
                "Escalar a Account Manager inmediatamente. "
                "Priorizar resolución en las próximas 4 horas."
            )
            
            if ticket_type == 'Correctivo':
                recommendations.append(
                    "→ Ticket correctivo crítico: Asignar al equipo senior. "
                    "Considerar llamada directa con el cliente para entender el impacto."
                )
            else:
                recommendations.append(
                    "→ Aunque es evolutivo, el alto riesgo sugiere frustración acumulada. "
                    "Revisar historial de incidentes del cliente."
                )
        
        elif risk_segment == 'Medio':
            recommendations.append(
                f"🟡 RIESGO MEDIO DE CHURN ({churn_risk:.0f}%): "
                "Monitorear de cerca. Resolver dentro de 24 horas."
            )
            
            if ticket_type == 'Correctivo':
                recommendations.append(
                    "→ Asignar al equipo de soporte técnico. "
                    "Mantener comunicación proactiva con el cliente."
                )
            else:
                recommendations.append(
                    "→ Evaluar viabilidad y estimar tiempo de desarrollo. "
                    "Responder en máximo 8 horas con roadmap."
                )
        
        else:  # Bajo
            recommendations.append(
                f"🟢 RIESGO BAJO DE CHURN ({churn_risk:.0f}%): "
                "Seguir proceso estándar de atención."
            )
            
            if ticket_type == 'Correctivo':
                recommendations.append(
                    "→ Resolver según SLA establecido. Documentar para base de conocimiento."
                )
            else:
                recommendations.append(
                    "→ Agregar a backlog de evolutivos. Priorizar según roadmap del producto."
                )
        
        # Unir todas las recomendaciones
        return "\n".join(recommendations)


class TicketRecommender:
    """
    Generador de recomendaciones para tickets.
    
    Implementa el principio de Responsabilidad Única (SRP) y
    depende de abstracciones (DIP).
    """
    
    def __init__(self, strategy: RecommendationStrategy):
        """
        Inicializa el recomendador con una estrategia.
        
        Args:
            strategy: Estrategia de recomendación a usar
        """
        self.strategy = strategy
    
    def generate_recommendation(self, ticket_data: Dict) -> str:
        """
        Genera recomendación usando la estrategia configurada.
        
        Args:
            ticket_data: Datos del ticket
            
        Returns:
            str: Texto con la recomendación
        """
        return self.strategy.recommend(ticket_data)
    
    def classify_risk_segment(self, churn_risk: float) -> str:
        """
        Clasifica el riesgo en segmentos.
        
        Args:
            churn_risk: Riesgo de churn (0-100)
            
        Returns:
            str: 'Bajo', 'Medio' o 'Alto'
        """
        if churn_risk >= 70:
            return 'Alto'
        elif churn_risk >= 40:
            return 'Medio'
        else:
            return 'Bajo'


# Factory para crear instancia por defecto
def create_ticket_recommender() -> TicketRecommender:
    """
    Crea una instancia de TicketRecommender con estrategia por defecto.
    
    Returns:
        TicketRecommender: Recomendador configurado y listo para usar
    """
    strategy = ChurnBasedRecommendation()
    return TicketRecommender(strategy)
