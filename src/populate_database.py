"""
Script para popular la base de datos con tickets de ejemplo.

Esto permite que las vistas de:
- 📊 Análisis de Datos
- 🔥 Factores de Churn

muestren información histórica basada en el dataset sintético.

Uso:
    python src/populate_database.py
"""

import sys
from pathlib import Path

import pandas as pd

# ----------------------------
# RUTAS DEL PROYECTO
# ----------------------------
# CURRENT_DIR = carpeta src/
CURRENT_DIR = Path(__file__).resolve().parent
# BASE_DIR = carpeta raíz del proyecto (hackathon-vortex-support-ai)
BASE_DIR = CURRENT_DIR.parent

# Asegurar que src/ esté en el path para los imports locales
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from db_utils import (
    insert_raw_ticket,
    insert_core_ticket,
    insert_gold_prediction,
)
from preprocessing import create_text_preprocessor
from security import create_security_analyzer
from nlp_pipeline import create_pipeline

# ✅ Rutas correctas
DATA_PATH = BASE_DIR / "data" / "tickets_train.csv"
MODELS_DIR = BASE_DIR / "models"


def populate_from_training_data(n_rows: int = 50) -> None:
    """
    Toma los primeros `n_rows` registros del CSV sintético
    y los pasa por TODO el pipeline:

    CSV -> raw_tickets -> core_tickets_enriched -> gold_ticket_predictions
    """
    if not DATA_PATH.exists():
        print(f"❌ No se encontró el archivo de datos: {DATA_PATH}")
        print("Tip: genera el dataset con:")
        print("   python src/generate_dummy_data.py")
        return

    if not MODELS_DIR.exists():
        print(f"❌ No se encontró la carpeta de modelos: {MODELS_DIR}")
        print("Tip: entrena los modelos con:")
        print("   python src/train_models.py")
        return

    # --- Cargar CSV ---
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        print("❌ El CSV no tiene filas.")
        return

    df = df.head(n_rows)

    # --- Instanciar componentes ---
    preprocessor = create_text_preprocessor()
    security_analyzer = create_security_analyzer()
    pipeline = create_pipeline(MODELS_DIR)

    print("=" * 60)
    print(f"🚀 Poblando la base de datos con {len(df)} tickets de ejemplo...")
    print("=" * 60)

    processed = 0

    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        client_name = row.get("client_name", "Cliente N/A")
        project_name = row.get("project_name", "Proyecto N/A")
        channel = row.get("channel", "email")
        project_age_days = int(row.get("project_age_days", 180))
        open_incidents_30d = int(row.get("open_incidents_30d", 2))

        # 1) RAW: guardar ticket original y obtener ticket_id
        ticket_id = insert_raw_ticket(
            client_name=client_name,
            project_name=project_name,
            channel=channel,
            original_text=text,
        )

        # 2) PREPROCESS + SECURITY (para CORE)
        pre = preprocessor.analyze(text)
        cleaned_text = pre["cleaned_text"]
        sentiment = float(pre["sentiment_score"])
        word_count = int(pre["word_count"])

        is_phishing, has_pii = security_analyzer.analyze(text)

        insert_core_ticket(
            ticket_id=ticket_id,
            cleaned_text=cleaned_text,
            is_phishing=bool(is_phishing),
            has_pii=bool(has_pii),
            sentiment_score=sentiment,
            word_count=word_count,
        )

        # 3) MODELOS (para GOLD)
        model_result = pipeline.process(
            text=text,
            project_age_days=project_age_days,
            open_incidents_30d=open_incidents_30d,
        )

        insert_gold_prediction(
            ticket_id=ticket_id,
            ticket_type_pred=model_result.ticket_type_pred,
            churn_risk_pred=model_result.churn_risk_pred,
            risk_segment=model_result.risk_segment,
            recommendation_text=model_result.recommendation_text,
        )

        processed += 1

    print()
    print("=" * 60)
    print(f"✅ Se procesaron {processed} tickets exitosamente.")
    print()
    print("Ahora las vistas de 'Análisis de Datos' y 'Factores de Churn'")
    print("mostrarán gráficos con datos reales.")
    print()
    print("🔄 Recarga el dashboard de Streamlit para ver los cambios.")
    print("   (Detén y vuelve a ejecutar `streamlit run src/dashboard_app.py` si es necesario)")
    print("=" * 60)


if __name__ == "__main__":
    # ✅ Usamos todos los registros del CSV (300)
    populate_from_training_data(300)
