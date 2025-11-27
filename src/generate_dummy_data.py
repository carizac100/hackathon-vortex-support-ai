"""
Generate synthetic ticket dataset for training and EDA.

Owner: R2 (Camilo) - Data generator for the whole team.
"""

import random
import csv
from pathlib import Path

DATA_PATH = Path("data/tickets_train.csv")

# Plantillas de texto para tickets correctivos y evolutivos
CORRECTIVE_TEMPLATES = [
    "Since the last deployment the invoicing module throws error 500 when generating invoices.",
    "The user cannot log in to the platform, it says invalid credentials even after password reset.",
    "The report page crashes when we filter by date range.",
    "The integration with the banking service stopped working and payments are not being processed.",
    "The system is extremely slow when we try to approve purchase orders."
]

EVOLUTIVE_TEMPLATES = [
    "We need a new dashboard to monitor monthly KPIs for our projects.",
    "The client is asking for an export to Excel for the contracts report.",
    "We would like to add a filter by project manager in the main view.",
    "We need a new API endpoint to read contract status from an external system.",
    "We want to schedule automatic email summaries of open incidents every Monday."
]

NEGATIVE_INTENSIFIERS = [
    "The client is very frustrated and angry about this issue.",
    "They say this situation is unacceptable and they are considering leaving the service.",
    "They have already complained several times and nothing has been solved.",
]

NEUTRAL_ADDITIONS = [
    "The client would like this to be prioritized next sprint.",
    "They mentioned this is important but not blocking production.",
    "They understand it may take some time, but want visibility on the roadmap.",
]

PHISHING_PHRASES = [
    "Please click here to reset your password immediately.",
    "Use this link to verify your account before it is deactivated.",
]

PII_EMAILS = [
    "You can contact me at john.doe@example.com.",
    "Please reply to maria.client@company.com with the solution.",
]

PII_PHONES = [
    "My phone number is 3112345678, call me as soon as possible.",
    "You can reach me at 3209876543 during office hours.",
]


def generate_single_row(ticket_id: int) -> dict:
    """
    Genera un ticket sintético con correlaciones razonables:
    - Correctivo suele tener churn más alto y sentimiento más negativo.
    - Más incidentes abiertos y proyectos más viejos → mayor churn.
    """
    # Tipo de ticket
    ticket_type = random.choice(["Correctivo", "Evolutivo"])

    # Proyecto y cliente
    client_name = f"Client {random.randint(1, 15)}"
    project_name = f"Project {random.randint(1, 8)}"
    channel = random.choice(["email", "portal", "whatsapp"])

    # Antigüedad del proyecto (1 mes a 2 años)
    project_age_days = random.randint(30, 730)

    # Incidentes abiertos en los últimos 30 días
    open_incidents_30d = random.randint(0, 15)

    # Texto base según el tipo
    if ticket_type == "Correctivo":
        text = random.choice(CORRECTIVE_TEMPLATES)
    else:
        text = random.choice(EVOLUTIVE_TEMPLATES)

    # Sentimiento y texto adicional
    if ticket_type == "Correctivo" and random.random() < 0.6:
        # más probable que estén molestos
        extra = random.choice(NEGATIVE_INTENSIFIERS)
        text = text + " " + extra
        sentiment_label = random.uniform(-0.9, -0.4)
    else:
        extra = random.choice(NEUTRAL_ADDITIONS)
        text = text + " " + extra
        sentiment_label = random.uniform(-0.1, 0.5)

    # Phishing
    if random.random() < 0.15:  # 15% de probabilidad
        text = text + " " + random.choice(PHISHING_PHRASES)
        is_phishing = 1
    else:
        is_phishing = 0

    # PII
    has_pii = 0
    if random.random() < 0.3:  # 30% de probabilidad de tener email o teléfono
        if random.random() < 0.5:
            text = text + " " + random.choice(PII_EMAILS)
        else:
            text = text + " " + random.choice(PII_PHONES)
        has_pii = 1

    # Cálculo del churn (0-100) con reglas simples pero coherentes
    # Base según tipo
    if ticket_type == "Correctivo":
        churn_risk = random.randint(40, 80)
    else:
        churn_risk = random.randint(10, 60)

    # Ajuste por número de incidentes
    churn_risk += open_incidents_30d * 2

    # Ajuste por antigüedad (proyectos muy viejos +10)
    if project_age_days > 365:
        churn_risk += 10

    # Ajuste por sentimiento
    if sentiment_label < -0.5:
        churn_risk += 15
    elif sentiment_label < 0:
        churn_risk += 5
    elif sentiment_label > 0.3:
        churn_risk -= 5

    # Ajuste si hay phishing (es muy grave)
    if is_phishing:
        churn_risk += 10

    # Limitar a [0, 100]
    churn_risk = max(0, min(100, churn_risk))

    return {
        "ticket_id": ticket_id,
        "client_name": client_name,
        "project_name": project_name,
        "channel": channel,
        "text": text,
        "ticket_type": ticket_type,
        "churn_risk": churn_risk,
        "project_age_days": project_age_days,
        "open_incidents_30d": open_incidents_30d,
        "sentiment_label": round(sentiment_label, 3),
        "is_phishing": is_phishing,
        "has_pii": has_pii,
    }


def generate_dataset(n_rows: int = 300) -> None:
    random.seed(42)

    rows = [generate_single_row(i + 1) for i in range(n_rows)]

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Synthetic dataset generated at {DATA_PATH.resolve()} with {n_rows} rows.")


if __name__ == "__main__":
    generate_dataset()
