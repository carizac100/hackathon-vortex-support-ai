"""
CLI script to train all ML models for Neuro Support AI.

It uses the high-level ModelTrainer defined in models.py and persists:

- models/ticket_classifier.pkl
- models/churn_predictor.pkl

Usage:
    python src/train_models.py
"""

from __future__ import annotations

from models import create_model_trainer


def main() -> None:
    trainer = create_model_trainer()

    print("🚀 Loading training data...")
    df = trainer.load_data()

    print("\n🔧 Training ticket type classifier (Correctivo/Evolutivo)...\n")
    _, clf_metrics = trainer.train_ticket_classifier(df)

    print("\n🔧 Training churn predictor (0–100)...\n")
    _, churn_metrics = trainer.train_churn_predictor(df)

    print("\n✅ Training completed.")
    print(f"Ticket classifier metrics: {clf_metrics}")
    print(f"Churn predictor metrics: {churn_metrics}")


if __name__ == "__main__":
    main()
