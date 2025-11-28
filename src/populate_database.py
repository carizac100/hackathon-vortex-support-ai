"""
Script para popular la base de datos con tickets de ejemplo.

Esto permite que las vistas de Análisis de Datos y Factores de Churn funcionen.

Ejecutar con: python3 src/populate_database.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from db_utils import insert_raw_ticket, insert_core_ticket, insert_gold_prediction
from nlp_pipeline import create_pipeline
from init_db import init_db


def populate_from_training_data(num_tickets: int = 50):
    """
    Popula la BD procesando tickets del dataset de entrenamiento.
    
    Args:
        num_tickets: Número de tickets a procesar (default: 50)
    """
    print("=" * 60)
    print("🗄️  POPULANDO BASE DE DATOS")
    print("=" * 60)
    print()
    
    # 1. Inicializar BD
    print("📊 Inicializando base de datos...")
    init_db()
    print()
    
    # 2. Cargar pipeline
    print("🤖 Cargando pipeline...")
    models_dir = Path("models")
    pipeline = create_pipeline(models_dir)
    print()
    
    # 3. Cargar datos de entrenamiento
    print(f"📂 Cargando {num_tickets} tickets del dataset de entrenamiento...")
    data_path = Path("data/tickets_train.csv")
    df = pd.read_csv(data_path).head(num_tickets)
    print(f"✅ {len(df)} tickets cargados\n")
    
    # 4. Procesar cada ticket
    print("🔄 Procesando tickets...")
    print()
    
    for idx, row in df.iterrows():
        try:
            # Insertar ticket raw
            ticket_id = insert_raw_ticket(
                client_name=row['client_name'],
                project_name=row['project_name'],
                channel=row['channel'],
                original_text=row['text']
            )
            
            # Procesar con pipeline
            result = pipeline.process(
                text=row['text'],
                project_age_days=int(row['project_age_days']),
                open_incidents_30d=int(row['open_incidents_30d'])
            )
            
            # Insertar en CORE
            insert_core_ticket(
                ticket_id=ticket_id,
                cleaned_text=result.cleaned_text,
                is_phishing=1 if result.is_phishing else 0,
                has_pii=1 if result.has_pii else 0,
                sentiment_score=result.sentiment_score,
                word_count=result.word_count
            )
            
            # Insertar en GOLD
            insert_gold_prediction(
                ticket_id=ticket_id,
                ticket_type_pred=result.ticket_type_pred,
                churn_risk_pred=result.churn_risk_pred,
                risk_segment=result.risk_segment,
                recommendation_text=result.recommendation_text
            )
            
            if (idx + 1) % 10 == 0:
                print(f"  ✅ Procesados {idx + 1}/{len(df)} tickets...")
        
        except Exception as e:
            print(f"  ⚠️  Error procesando ticket {idx + 1}: {e}")
            continue
    
    print()
    print("=" * 60)
    print("✅ BASE DE DATOS POPULADA")
    print("=" * 60)
    print()
    print(f"Se procesaron {len(df)} tickets exitosamente.")
    print()
    print("Ahora las vistas de 'Análisis de Datos' y 'Factores de Churn'")
    print("mostrarán gráficos con datos reales.")
    print()
    print("🚀 Recarga el dashboard para ver los cambios:")
    print("   http://localhost:8501")
    print()


if __name__ == "__main__":
    populate_from_training_data(50)
