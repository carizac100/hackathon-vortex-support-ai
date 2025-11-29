"""
Generador de dataset sintético de tickets de soporte EN ESPAÑOL
para Neuro Support AI (Black_Cyber).

Salida:
    data/tickets_train.csv

Columnas principales:
    ticket_id
    client_name
    project_name
    channel
    text
    ticket_type
    churn_risk
    project_age_days
    open_incidents_30d
    sentiment_label
    is_phishing
    has_pii
"""

from pathlib import Path
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


N_ROWS = 300
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "tickets_train.csv"


# -----------------------
# Catálogos en ESPAÑOL
# -----------------------

CLIENTES = [
    "Cliente Andes",
    "Cliente Pacífico",
    "Cliente Caribe",
    "Banco Aurora",
    "Retail Nova",
    "Finanzas Alpha",
    "Logística Sur",
    "TecnoCloud",
    "SaludNet",
    "EduTech Global",
]

PROYECTOS = [
    "Portal de Pagos",
    "Módulo de Facturación",
    "App de Soporte",
    "Tablero de KPIs",
    "Portal de Proveedores",
    "BI Ejecutivo",
    "Integración ERP",
    "App Móvil Clientes",
    "Gestión de Contratos",
    "API de Notificaciones",
]

CANALES = ["correo", "portal", "whatsapp"]

TEXTOS_CORRECTIVOS = [
    "Desde la última actualización el módulo de facturación devuelve error 500 al generar las facturas.",
    "La pantalla de reportes se queda en blanco cuando filtramos por rango de fechas.",
    "Los usuarios no pueden iniciar sesión, el sistema muestra credenciales inválidas aunque son correctas.",
    "El botón de exportar a Excel no descarga el archivo y no muestra mensaje de error.",
    "El portal está muy lento, las páginas tardan más de 10 segundos en cargar.",
    "Al aprobar una orden de compra, el sistema se queda congelado y debemos reiniciar.",
    "Los correos de recuperación de contraseña no están llegando a los clientes.",
    "El menú principal desaparece cuando se abre el módulo de contratos.",
    "Los datos de ventas diarios no se actualizan desde hace dos días.",
    "El dashboard de indicadores muestra totales diferentes a los del reporte detallado.",
]

TEXTOS_EVOLUTIVOS = [
    "El cliente solicita agregar un filtro por sucursal en el tablero de KPIs para segmentar mejor la información.",
    "Nos piden incluir un resumen ejecutivo con gráficos en el informe mensual descargable en PDF.",
    "Requieren integrar el módulo de facturación con un nuevo proveedor de pagos en línea.",
    "El equipo comercial solicita una vista móvil optimizada para revisar métricas desde el celular.",
    "Piden agregar un campo de referencia interna en las órdenes de compra para rastrear campañas.",
    "El cliente quiere que el sistema envíe alertas automáticas cuando el SLA esté cerca de incumplirse.",
    "Nos solicitan habilitar autenticación con doble factor para los usuarios administradores.",
    "El área de finanzas pide poder exportar los reportes también en formato CSV.",
    "Plantean la creación de un panel específico para seguimiento de tickets críticos.",
    "El cliente propone un módulo de autoservicio donde ellos mismos puedan configurar algunos reportes.",
]

TEXTOS_PHISHING = [
    "Recibimos un correo que solicita actualizar las credenciales en un enlace sospechoso, parece un intento de phishing.",
    "Un usuario reporta que le llegó un mensaje pidiendo datos de tarjeta de crédito para mantener el acceso a la plataforma.",
]

# Palabras que refuerzan sentimiento negativo / positivo
NEGATIVAS = ["urgente", "crítico", "inaceptable", "muy molesto", "frustrado"]
POSITIVAS = ["gracias", "excelente", "rápido", "muy contentos", "funciona bien"]


def generar_ticket(i: int) -> dict:
    """Genera un ticket sintético en español."""
    client = random.choice(CLIENTES)
    project = random.choice(PROYECTOS)
    channel = random.choice(CANALES)

    # Elegir tipo de ticket
    ticket_type = random.choices(
        ["Correctivo", "Evolutivo"],
        weights=[0.6, 0.4],
        k=1,
    )[0]

    # Base de texto según tipo
    if ticket_type == "Correctivo":
        text = random.choice(TEXTOS_CORRECTIVOS)
    else:
        text = random.choice(TEXTOS_EVOLUTIVOS)

    # Inyectar algunos casos de phishing / PII
    is_phishing = 0
    has_pii = 0

    # ~5% de tickets con phishing explícito
    if random.random() < 0.05:
        text += " Además, el usuario adjunta el enlace sospechoso que pide usuario y contraseña."
        text = random.choice(TEXTOS_PHISHING) + " " + text
        is_phishing = 1

    # ~10% con PII (correo / teléfono / cédula falsa)
    if random.random() < 0.1:
        text += " El cliente deja su correo juan.perez@example.com y el número 3201234567 para contacto."
        has_pii = 1

    # Ajustar sentimiento simple
    sentimiento = 0  # neutro
    if ticket_type == "Correctivo":
        if random.random() < 0.7:
            text += " El cliente está muy molesto y necesita solución urgente."
            text += " " + random.choice(NEGATIVAS)
            sentimiento = -1
    else:
        if random.random() < 0.5:
            text += " Comentan que en general están muy contentos con la plataforma."
            text += " " + random.choice(POSITIVAS)
            sentimiento = 1

    # Edad del proyecto e incidentes
    project_age_days = random.randint(30, 540)          # entre 1 y 18 meses
    open_incidents_30d = random.randint(0, 10)

    # Riesgo de churn aproximado (regla heurística)
    base_churn = 20
    base_churn += max(0, open_incidents_30d - 3) * 5
    if ticket_type == "Correctivo":
        base_churn += 10
    if sentimiento == -1:
        base_churn += 20
    if is_phishing:
        base_churn += 5

    churn_risk = max(0, min(100, base_churn + random.randint(-5, 5)))

    created_at = datetime.now() - timedelta(days=random.randint(0, 60))

    return {
        "ticket_id": i,
        "client_name": client,
        "project_name": project,
        "channel": channel,
        "text": text,
        "ticket_type": ticket_type,
        "churn_risk": churn_risk,
        "project_age_days": project_age_days,
        "open_incidents_30d": open_incidents_30d,
        "sentiment_label": sentimiento,
        "is_phishing": is_phishing,
        "has_pii": has_pii,
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    registros = [generar_ticket(i + 1) for i in range(N_ROWS)]
    df = pd.DataFrame(registros)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✅ Dataset sintético en ESPAÑOL generado en: {OUTPUT_PATH.resolve()}")
    print(f"   Filas: {len(df)}")


if __name__ == "__main__":
    main()
