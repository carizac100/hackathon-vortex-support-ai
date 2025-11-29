"""
CLI script to train all ML models used by Neuro Support AI.

Usage:
    python -m src.train_models
    or
    python src/train_models.py
"""

import sys
from pathlib import Path

# Ensure local src/ is importable when running as script
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models import create_model_trainer


def main() -> None:
    trainer = create_model_trainer()

    try:
        print("🚀 Loading training data...")
        df = trainer.load_data()

        print("\n🔧 Training ticket type classifier (Correctivo/Evolutivo)...")
        _, cls_metrics = trainer.train_ticket_classifier(df)

        print("\n🔧 Training churn predictor (0–100)...")
        _, churn_metrics = trainer.train_churn_predictor(df)

        print("\n✅ Training completed.")
        print("Ticket classifier metrics:", cls_metrics)
        print("Churn predictor metrics:", churn_metrics)

    except FileNotFoundError as e:
        print(f"\n❌ Dataset not found: {e}")
        print("\nTip: generate it first with:")
        print("   python src/generate_dummy_data.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
