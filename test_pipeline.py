"""
Script de prueba rápida para debuggear el pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nlp_pipeline import create_pipeline

def test_pipeline():
    """Prueba el pipeline con un texto simple."""
    
    # Cargar pipeline
    models_dir = Path("models")
    print(f"📂 Cargando modelos desde: {models_dir.absolute()}")
    
    try:
        pipeline = create_pipeline(models_dir)
        print("✅ Pipeline cargado correctamente\n")
        
        # Texto de prueba
        test_text = "El sistema está muy lento y el cliente está furioso"
        print(f"📝 Texto de prueba: {test_text}\n")
        
        # Procesar
        result = pipeline.process(
            text=test_text,
            project_age_days=180,
            open_incidents_30d=2
        )
        
        print("✅ RESULTADO:")
        print(f"  ├─ Tipo: {result.ticket_type_pred}")
        print(f"  ├─ Churn: {result.churn_risk_pred:.1f}%")
        print(f"  ├─ Segmento: {result.risk_segment}")
        print(f"  ├─ Sentimiento: {result.sentiment_score}")
        print(f"  ├─ Phishing: {result.is_phishing}")
        print(f"  └─ PII: {result.has_pii}")
        print()
        print(f"💡 Recomendación:\n{result.recommendation_text}")
        
    except Exception as e:
        import traceback
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        print("\nTraceback completo:")
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
