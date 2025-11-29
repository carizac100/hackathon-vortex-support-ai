# dashboard_app.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go  # por si luego lo necesitas
import base64

sys.path.insert(0, str(Path(__file__).parent))

from nlp_pipeline import create_pipeline
from db_utils import (
    get_connection,
    insert_raw_ticket,
    insert_core_ticket,
    insert_gold_prediction,
)
from preprocessing import sentiment_score, clean_text
from generate_dummy_data import CLIENTES
from security import detect_pii, mask_pii, detect_phishing, detect_aggressive_language


st.set_page_config(
    page_title="Neuro Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# TEMA (LIGHT / DARK) EN SESIÓN
# ===========================
if "tema" not in st.session_state:
    st.session_state["tema"] = "oscuro"  # valores: "oscuro" / "claro"

tema_actual = st.session_state["tema"]

# ===========================
# CSS GLOBAL (MODO OSCURO / CLARO)
# ===========================
if tema_actual == "oscuro":
    css_vars = """
    :root {
        --ns-bg: #050816;
        --ns-bg-elevated: #0b1220;
        --ns-bg-soft: #020617;

        --ns-primary: #6366f1;
        --ns-primary-strong: #4f46e5;
        --ns-primary-soft: rgba(99, 102, 241, 0.16);

        --ns-accent: #22c55e;

        --ns-text-main: #e5e7eb;
        --ns-text-muted: #9ca3af;

        --ns-border-subtle: rgba(148, 163, 184, 0.35);

        --ns-danger: #f97373;
        --ns-danger-soft: rgba(248, 113, 113, 0.12);

        --ns-warning: #facc15;
        --ns-warning-soft: rgba(250, 204, 21, 0.10);

        --ns-success: #22c55e;
        --ns-success-soft: rgba(34, 197, 94, 0.15);

        --ns-sidebar-bg1: #020617;
        --ns-sidebar-bg2: #020617;
        --ns-sidebar-border: rgba(15,23,42,0.9);

        --ns-bg-grad-1: #020617;
        --ns-bg-grad-2: #020617;
        --ns-bg-grad-3: #111827;
    }
    """
else:
    css_vars = """
    :root {
        --ns-bg: #e5e7eb;
        --ns-bg-elevated: #ffffff;
        --ns-bg-soft: #eff6ff;

        --ns-primary: #4f46e5;
        --ns-primary-strong: #4338ca;
        --ns-primary-soft: rgba(79, 70, 229, 0.12);

        --ns-accent: #16a34a;

        --ns-text-main: #0f172a;
        --ns-text-muted: #6b7280;

        --ns-border-subtle: rgba(148, 163, 184, 0.55);

        --ns-danger: #b91c1c;
        --ns-danger-soft: rgba(239, 68, 68, 0.08);

        --ns-warning: #b45309;
        --ns-warning-soft: rgba(251, 191, 36, 0.12);

        --ns-success: #15803d;
        --ns-success-soft: rgba(22, 163, 74, 0.12);

        --ns-sidebar-bg1: #eef2ff;
        --ns-sidebar-bg2: #f9fafb;
        --ns-sidebar-border: rgba(148,163,184,0.7);

        --ns-bg-grad-1: #e5e7eb;
        --ns-bg-grad-2: #eef2ff;
        --ns-bg-grad-3: #dbeafe;
    }
    """

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ======== Variables de tema ======== */
{css_vars}

/* ======== Base tipográfica ======== */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--ns-text-main);
}}

/* ======== Fondo global con degradado ======== */
.stApp {{
    background:
      radial-gradient(circle at top left, rgba(79, 70, 229, 0.45), transparent 55%),
      radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.35), transparent 55%),
      linear-gradient(135deg, var(--ns-bg-grad-1) 0%, var(--ns-bg-grad-2) 40%, var(--ns-bg-grad-3) 100%);
    color: var(--ns-text-main);
}}

/* Card central tipo Berry */
.block-container {{
    max-width: 1240px;
    margin: 1.5rem auto 3rem auto;
    padding: 1.5rem 1.8rem 2.5rem 1.8rem;
    background: var(--ns-bg-elevated);
    border-radius: 22px;
    border: 1px solid var(--ns-border-subtle);
    box-shadow: 0 30px 80px rgba(15, 23, 42, 0.25);
}}

/* Ocultar la franja de Streamlit arriba */
[data-testid="stDecoration"] {{
    display: none;
}}

/* ======== SIDEBAR TIPO BERRY ======== */
[data-testid="stSidebar"] {{
    background: radial-gradient(circle at top, var(--ns-sidebar-bg1) 0%, var(--ns-sidebar-bg2) 55%);
    border-right: 1px solid var(--ns-sidebar-border);
}}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: var(--ns-primary) !important;
}}

/* Logo circular en sidebar */
[data-testid="stSidebar"] img {{
    border-radius: 999px !important;
    border: 2px solid rgba(79, 70, 229, 0.6);
}}

/* Radios de Tema / Navegación con estilo menú */
.stRadio > div[role="radiogroup"] > label {{
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.15rem;
    cursor: pointer;
    transition: background 0.18s ease, color 0.18s ease;
}}

.stRadio > div[role="radiogroup"] > label:hover {{
    background: rgba(148, 163, 184, 0.18);
}}

.stRadio label p {{
    margin: 0;
    color: var(--ns-text-main) !important;
    font-size: 0.92rem;
}}

.stRadio input[type="radio"] {{
    accent-color: var(--ns-primary-strong);
}}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stHeader"] svg {{
    fill: var(--ns-text-main) !important;
}}

/* ======== HEADER PRINCIPAL ======== */
.neuro-header {{
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 18px 22px;
    margin: 0 0 1.8rem 0;

    background: linear-gradient(120deg, rgba(79, 70, 229, 0.28), rgba(14, 165, 233, 0.18));
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.45);
    box-shadow: 0 20px 55px rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(16px);
}}

.neuro-logo {{
    width: 72px;
    height: 72px;
    border-radius: 999px;
    object-fit: cover;
    border: 2px solid var(--ns-primary);
    box-shadow: 0 0 18px rgba(79, 70, 229, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    animation: neuro-pulse 3s ease-in-out infinite;
}}

.neuro-title {{
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a855f7, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 3px;
}}

.neuro-sub {{
    font-size: 0.93rem;
    color: var(--ns-text-muted);
}}

.neuro-badge {{
    margin-left: auto;
    background: var(--ns-primary-soft);
    border: 1px solid var(--ns-primary);
    color: #e0f2fe;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ======== TIPOGRAFÍA Y TÍTULOS ======== */
h1, h2, h3, h4 {{
    color: var(--ns-text-main);
    font-weight: 600;
}}

.view-header {{
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 0 1.1rem 0;
    background: linear-gradient(90deg, #a855f7, var(--ns-primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 14px rgba(129,140,248,0.35);
}}

.stMarkdown p, .stSidebar .stMarkdown p {{
    color: var(--ns-text-main) !important;
    font-size: 0.95rem;
}}

small, .stCaption, .stMarkdown small {{
    color: var(--ns-text-muted) !important;
    font-size: 0.8rem;
}}

/* ======== CARDS / MÉTRICAS ======== */
.metric-card {{
    background: var(--ns-bg-soft);
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid var(--ns-border-subtle);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    border-color: var(--ns-primary);
    box-shadow: 0 12px 25px rgba(15, 23, 42, 0.24);
}}

[data-testid="stMetric"] {{
    background: var(--ns-bg-soft);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    border: 1px solid var(--ns-border-subtle);
}}

[data-testid="stMetricLabel"] {{
    color: var(--ns-text-muted) !important;
    font-size: 0.85rem;
}}

[data-testid="stMetricValue"] {{
    color: var(--ns-text-main) !important;
    font-size: 1.1rem;
    font-weight: 600;
}}

/* ======== BOTONES ======== */
.stButton > button {{
    background: linear-gradient(90deg, var(--ns-primary-strong), var(--ns-primary));
    color: #FFFFFF;
    border: none;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.4rem;
    transition: box-shadow 0.18s ease, transform 0.18s ease, filter 0.18s ease;
}}

.stButton > button:hover {{
    box-shadow: 0 0 22px rgba(79, 70, 229, 0.7);
    transform: translateY(-1px);
    filter: brightness(1.05);
}}

.stDownloadButton > button {{
    border-radius: 999px;
    border: 1px solid var(--ns-primary-soft);
    background: rgba(15, 23, 42, 0.06);
    color: var(--ns-text-main);
}}

/* ======== INPUTS ======== */
.stTextInput > div > input,
.stNumberInput input,
.stDateInput input {{
    background: rgba(15, 23, 42, 0.08) !important;
    border-radius: 8px !important;
    color: var(--ns-text-main) !important;
    border: 1px solid rgba(148, 163, 184, 0.7) !important;
}}

.stTextArea textarea {{
    background: rgba(15, 23, 42, 0.08) !important;
    color: var(--ns-text-main) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(148, 163, 184, 0.7) !important;
}}

.stTextArea label,
.stNumberInput label,
.stTextInput label,
.stSelectbox label,
.stDateInput label,
.stMultiSelect label {{
    color: var(--ns-text-main) !important;
    font-weight: 600;
    font-size: 0.95rem;
}}

::placeholder {{
    color: rgba(148, 163, 184, 0.8) !important;
    opacity: 1;
}}

/* ======== ALERTAS ======== */
.warning-box {{
    background-color: var(--ns-warning-soft);
    border-left: 4px solid var(--ns-warning);
    padding: 0.9rem 1rem;
    border-radius: 10px;
}}

.danger-box {{
    background-color: var(--ns-danger-soft);
    border-left: 4px solid var(--ns-danger);
    padding: 0.9rem 1rem;
    border-radius: 10px;
}}

.success-box {{
    background-color: var(--ns-success-soft);
    border-left: 4px solid var(--ns-success);
    padding: 0.9rem 1rem;
    border-radius: 10px;
}}

.block-container .stAlert {{
    border-radius: 10px;
}}

/* ======== TABS ======== */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: rgba(15, 23, 42, 0.08);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.9rem;
    color: var(--ns-text-muted);
}}

.stTabs [aria-selected="true"] {{
    background: var(--ns-primary-soft);
    color: #0f172a !important;
    border: 1px solid var(--ns-primary);
}}

/* ======== FOOTER ======== */
.footer {{
    margin-top: 2.5rem;
    text-align: center;
    font-size: 0.8rem;
    color: var(--ns-text-muted);
}}

/* ======== ANIMACIÓN LOGO ======== */
@keyframes neuro-pulse {{
    0% {{ transform: scale(1); box-shadow: 0 0 0 rgba(79, 70, 229, 0.0); }}
    50% {{ transform: scale(1.03); box-shadow: 0 0 22px rgba(79, 70, 229, 0.9); }}
    100% {{ transform: scale(1); box-shadow: 0 0 0 rgba(79, 70, 229, 0.0); }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ===========================
# HEADER GLOBAL
# ===========================
def render_main_header():
    """Header global de Neuro Support AI (se muestra en todas las vistas)."""
    logo_path = Path("assets/logo.jpeg")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{encoded}" class="neuro-logo">'
    else:
        logo_html = '<div class="neuro-logo">🧠</div>'

    st.markdown(
        f"""
    <div class="neuro-header">
        {logo_html}
        <div>
            <div class="neuro-title">Neuro Support AI</div>
            <div class="neuro-sub">
                Análisis inteligente de tickets, seguridad y predicción de churn
            </div>
        </div>
        <div class="neuro-badge">BETA</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_pipeline():
    """Carga el pipeline de NLP con modelos entrenados."""
    current_file = Path(__file__).parent
    models_dir = current_file.parent / "models"

    if not models_dir.exists() or not (models_dir / "ticket_classifier.pkl").exists():
        st.error(f"⚠️ Los modelos no están entrenados. Ruta buscada: {models_dir}")
        st.error("Ejecuta primero: python3 src/train_models.py")
        st.stop()

    return create_pipeline(models_dir)


@st.cache_data
def load_gold_data():
    """Carga datos de la tabla GOLD para análisis."""
    try:
        conn = get_connection()
        query = """
        SELECT 
            g.ticket_id,
            r.client_name,
            r.project_name,
            r.channel,
            r.created_at,
            c.sentiment_score,
            c.is_phishing,
            c.has_pii,
            c.word_count,
            g.ticket_type_pred,
            g.churn_risk_pred,
            g.risk_segment
        FROM gold_ticket_predictions g
        JOIN raw_tickets r ON g.ticket_id = r.ticket_id
        JOIN core_tickets_enriched c ON g.ticket_id = c.ticket_id
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ===========================
# VISTA 1: ANÁLISIS DE TICKET
# ===========================
def view_ticket_analyzer(pipeline):
    """Vista 1: Formulario para analizar tickets individuales."""
    render_main_header()

    st.markdown(
        "### 📝 1. Ticket y contexto\n"
        "**Pega el texto del ticket** y completa la información del proyecto para obtener un análisis con IA:"
    )
    st.markdown(
        """
    - 🔒 Detección de seguridad (phishing, PII, lenguaje agresivo)  
    - 📊 Clasificación del tipo de ticket  
    - 📉 Predicción de riesgo de churn  
    - 💡 Recomendaciones de acción para el equipo  
    """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        ticket_text = st.text_area(
            "Texto del Ticket:",
            height=220,
            placeholder=(
                "Ejemplo: El módulo de facturación está lanzando error 500. "
                "El cliente está muy molesto y quiere una solución urgente..."
            ),
        )

    with col2:
        st.markdown("**Información del Proyecto**")
        st.caption("Estos datos ayudan al modelo a contextualizar el riesgo de churn.")

        client_name = st.selectbox(
            "Empresa / Cliente:",
            options=CLIENTES,
            index=0,
            help="Empresa a la que está asociado este proyecto.",
        )

        project_age = st.number_input(
            "Antigüedad del proyecto (días):",
            min_value=1,
            max_value=3650,
            value=180,
            help="¿Hace cuántos días se inició el proyecto?",
        )

        open_incidents = st.number_input(
            "Incidentes abiertos (últimos 30 días):",
            min_value=0,
            max_value=100,
            value=2,
            help="Número de tickets abiertos en los últimos 30 días.",
        )

    st.markdown("")

    if st.button("🔍 Analizar Ticket", type="primary", use_container_width=True):
        if not ticket_text or not ticket_text.strip():
            st.warning("⚠️ Por favor ingresa el texto del ticket.")
            return

        with st.spinner("Analizando ticket..."):
            try:
                model_result = {}
                try:
                    model_result = pipeline.process(
                        text=ticket_text,
                        project_age_days=int(project_age),
                        open_incidents_30d=int(open_incidents),
                    )
                except Exception:
                    model_result = None

                # Seguridad
                pii = detect_pii(ticket_text)
                masked = mask_pii(ticket_text)
                phishing = detect_phishing(ticket_text)
                aggressive = detect_aggressive_language(ticket_text)

                phishing_error = None
                is_phishing_flag = False
                phishing_indicators = []

                if isinstance(phishing, dict):
                    if "ERROR" in phishing:
                        phishing_error = phishing["ERROR"].get(
                            "message", "Error en analizador de phishing"
                        )
                        is_phishing_flag = False
                    else:
                        is_phishing_flag = bool(phishing.get("is_phishing", False))
                        phishing_indicators = (
                            phishing.get("found") or phishing.get("indicators") or []
                        )
                else:
                    is_phishing_flag = bool(phishing)

                has_pii_flag = False
                if isinstance(pii, dict):
                    try:
                        has_pii_flag = any(len(v) > 0 for v in pii.values())
                    except Exception:
                        has_pii_flag = False
                else:
                    try:
                        has_pii_flag = bool(pii)
                    except Exception:
                        has_pii_flag = False

                sentiment = sentiment_score(ticket_text)
                cleaned = clean_text(ticket_text)
                word_count = len(cleaned.split())

                ticket_type = (
                    getattr(model_result, "ticket_type_pred", "No disponible")
                    if model_result
                    else "No disponible"
                )
                churn_pred = (
                    getattr(model_result, "churn_risk_pred", np.nan)
                    if model_result
                    else np.nan
                )
                risk_segment = (
                    getattr(model_result, "risk_segment", "No disponible")
                    if model_result
                    else "No disponible"
                )
                recommendation_text = (
                    getattr(
                        model_result,
                        "recommendation_text",
                        "No hay recomendaciones del modelo. Revisa manualmente.",
                    )
                    if model_result
                    else "No hay recomendaciones del modelo. Revisa manualmente."
                )
                cleaned_text_from_model = (
                    getattr(model_result, "cleaned_text", cleaned)
                    if model_result
                    else cleaned
                )

                # Guardar en BD
                try:
                    ticket_id = insert_raw_ticket(
                        client_name=client_name,
                        project_name="Proyecto actual",
                        channel="Plataforma",
                        original_text=ticket_text,
                    )

                    insert_core_ticket(
                        ticket_id=ticket_id,
                        cleaned_text=cleaned_text_from_model,
                        is_phishing=bool(is_phishing_flag),
                        has_pii=bool(has_pii_flag),
                        sentiment_score=float(sentiment),
                        word_count=int(word_count),
                    )

                    if model_result is not None:
                        churn_value = (
                            float(churn_pred) if not pd.isna(churn_pred) else 0.0
                        )
                        insert_gold_prediction(
                            ticket_id=ticket_id,
                            ticket_type_pred=str(ticket_type),
                            churn_risk_pred=churn_value,
                            risk_segment=str(risk_segment),
                            recommendation_text=recommendation_text,
                        )

                    st.success(
                        f"💾 Ticket guardado con ID #{ticket_id} para el cliente **{client_name}**."
                    )
                except Exception as db_err:
                    st.warning(
                        f"⚠️ El ticket se analizó, pero no se pudo guardar en la base de datos: {db_err}"
                    )

                st.markdown("---")
                st.markdown("### 📋 2. Resumen del análisis")

                churn_display = (
                    "No disponible" if np.isnan(churn_pred) else f"{churn_pred:.1f}%"
                )

                sentiment_emoji = (
                    "😠"
                    if sentiment < -0.3
                    else ("😐" if sentiment < 0.3 else "😊")
                )
                sentiment_label = (
                    "Negativo"
                    if sentiment < -0.3
                    else ("Neutral" if sentiment < 0.3 else "Positivo")
                )
                risk_color = (
                    "🔴"
                    if risk_segment == "Alto"
                    else ("🟡" if risk_segment == "Medio" else "🟢")
                )

                st.markdown(
                    f"En este ticket se detecta **riesgo de churn {risk_segment}** "
                    f"({churn_display}), con un **sentimiento {sentiment_label.lower()}** "
                    f"({sentiment:.2f}) y "
                    f"{'alertas de seguridad activas' if (is_phishing_flag or has_pii_flag or aggressive) else 'sin alertas de seguridad críticas'}."
                )

                if is_phishing_flag or has_pii_flag or aggressive:
                    st.markdown(
                        '<div class="danger-box">', unsafe_allow_html=True
                    )
                    st.markdown("#### 🚨 Alertas de seguridad")
                    if is_phishing_flag:
                        st.error(
                            "⚠️ **PHISHING DETECTADO**: Este ticket contiene patrones sospechosos de phishing."
                        )
                    if has_pii_flag:
                        st.warning(
                            "⚠️ **PII DETECTADO**: El ticket contiene información personal identificable."
                        )
                    if aggressive:
                        st.error(
                            "😡 **LENGUAJE AGRESIVO DETECTADO**: "
                            "El cliente utiliza un tono ofensivo o inapropiado."
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                col1m, col2m, col3m, col4m = st.columns(4)

                with col1m:
                    st.metric(
                        "Tipo de Ticket",
                        ticket_type,
                        delta="Correctivo"
                        if ticket_type == "Correctivo"
                        else "Evolutivo",
                        delta_color="inverse"
                        if ticket_type == "Correctivo"
                        else "normal",
                    )

                with col2m:
                    st.metric(
                        "Riesgo de Churn",
                        churn_display,
                        delta=f"{risk_color} {risk_segment}",
                    )

                with col3m:
                    st.metric(
                        "Puntaje de Sentimiento",
                        value=f"{sentiment:.2f}",
                        delta=f"{sentiment_emoji}  {sentiment_label}",
                    )

                with col4m:
                    st.metric("Palabras en el ticket", word_count)

                st.markdown("---")
                st.markdown("### 💡 3. Recomendaciones de acción")

                if aggressive:
                    st.error(
                        "😡 **Protocolo de conducta:** Se detectó lenguaje agresivo. "
                        "Aplica el protocolo de escalamiento de comportamiento. "
                        "Mantén la calma y usa un tono profesional y empático para desescalar."
                    )

                if risk_segment == "Alto":
                    st.markdown(
                        '<div class="danger-box">', unsafe_allow_html=True
                    )
                elif risk_segment == "Medio":
                    st.markdown(
                        '<div class="warning-box">', unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="success-box">', unsafe_allow_html=True
                    )

                st.markdown(recommendation_text.replace("\\n", "\\n\\n"))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 🔎 4. Detalle del análisis")

                tab1, tab2, tab3 = st.tabs(
                    [
                        "Resumen ejecutivo",
                        "Seguridad (PII, phishing, tono)",
                        "Texto y preprocesamiento",
                    ]
                )

                with tab1:
                    st.markdown("#### ✅ Resumen general")
                    st.write("- **Tipo de ticket:** ", ticket_type)
                    st.write(
                        f"- **Riesgo de churn:** {churn_display} ({risk_segment})"
                    )
                    st.write(
                        f"- **Sentimiento (score):** {sentiment:.3f} — {sentiment_label}"
                    )
                    st.write(f"- **Longitud (palabras):** {word_count}")
                    st.write(
                        f"- **Phishing detectado:** {'Sí' if is_phishing_flag else 'No'}"
                    )
                    st.write(
                        f"- **Lenguaje agresivo:** {'Sí' if aggressive else 'No'}"
                    )
                    st.write(f"- **PII detectada:** {'Sí' if has_pii_flag else 'No'}")
                    if is_phishing_flag and phishing_indicators:
                        st.write(
                            f"- **Indicadores de phishing encontrados:** "
                            f"{', '.join(phishing_indicators)}"
                        )

                with tab2:
                    st.markdown("#### 🔒 PII detectada")
                    st.caption(
                        "Resumen estructurado de la PII encontrada."
                    )
                    with st.expander(
                        "Ver detalle estructurado de PII (JSON)"
                    ):
                        st.json(pii)

                    st.markdown("#### 🔐 Texto enmascarado")
                    st.text_area(
                        "Texto con PII enmascarada:",
                        masked,
                        height=150,
                        disabled=True,
                    )

                    st.markdown("#### 🐟 Resultado de phishing")
                    if phishing_error:
                        st.info(
                            "No se pudo ejecutar el análisis avanzado de phishing en este ticket. "
                            "Para este caso asumimos que **no es phishing**."
                        )
                        st.code(phishing_error)
                    else:
                        st.json(phishing)

                    st.markdown("#### 😡 Lenguaje agresivo")
                    if aggressive:
                        st.error(
                            "El detector encontró patrones de lenguaje inapropiado o agresivo. "
                            "**Acción requerida: aplicar protocolo de conducta.**"
                        )
                    else:
                        st.success(
                            "No se detectó lenguaje agresivo en este ticket."
                        )

                with tab3:
                    st.markdown("#### 🧹 Texto limpio / normalizado")
                    st.text_area(
                        "Texto limpio (lowercase, normalizado):",
                        cleaned_text_from_model,
                        height=120,
                        disabled=True,
                    )

                    st.markdown("#### 📝 Texto original")
                    st.text_area(
                        "Texto original:", ticket_text, height=150, disabled=True
                    )

                    st.markdown("#### 💬 Sentimiento (valor numérico)")
                    st.write(sentiment)

            except Exception as e:
                import traceback
                st.error(
                    f"❌ Error al procesar el ticket: {type(e).__name__}: {str(e)}"
                )
                st.code(traceback.format_exc())


# ===========================
# VISTA 2: ANÁLISIS DE DATOS
# ===========================
def view_data_analytics():
    """Vista 2: Gráficos y análisis de datos."""
    render_main_header()
    st.markdown(
        '<div class="view-header">📊 Análisis de Datos</div>',
        unsafe_allow_html=True,
    )

    try:
        df = load_gold_data()

        if df.empty:
            st.info(
                "ℹ️ No hay datos históricos aún. Procesa algunos tickets primero."
            )
            return

        if df["created_at"].dtype == "object":
            df["created_at"] = pd.to_datetime(df["created_at"])

        st.markdown("### 🔍 Filtros")

        f1_col1, f1_col2 = st.columns([1.2, 1])

        with f1_col1:
            min_date = df["created_at"].min().date()
            max_date = df["created_at"].max().date()

            date_range = st.date_input(
                "Rango de fechas:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filtra tickets por fecha de creación.",
            )

        with f1_col2:
            all_clients = sorted(df["client_name"].unique().tolist())
            selected_clients = st.multiselect(
                "Clientes:",
                options=all_clients,
                default=all_clients,
                help="Selecciona uno o más clientes.",
            )

        f2_col1, f2_col2 = st.columns(2)

        with f2_col1:
            all_channels = sorted(df["channel"].unique().tolist())
            selected_channels = st.multiselect(
                "Canales:",
                options=all_channels,
                default=all_channels,
                help="Filtra por canal de comunicación.",
            )

        with f2_col2:
            all_segments = ["Bajo", "Medio", "Alto"]
            selected_segments = st.multiselect(
                "Segmento de Riesgo:",
                options=all_segments,
                default=all_segments,
                help="Filtra por nivel de riesgo de churn.",
            )

        df_filtered = df.copy()

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_filtered[
                (df_filtered["created_at"].dt.date >= start_date)
                & (df_filtered["created_at"].dt.date <= end_date)
            ]

        if selected_clients:
            df_filtered = df_filtered[
                df_filtered["client_name"].isin(selected_clients)
            ]

        if selected_channels:
            df_filtered = df_filtered[
                df_filtered["channel"].isin(selected_channels)
            ]

        if selected_segments:
            df_filtered = df_filtered[
                df_filtered["risk_segment"].isin(selected_segments)
            ]

        if len(df_filtered) < len(df):
            st.info(
                f"📊 Mostrando {len(df_filtered)} de {len(df)} tickets "
                f"({len(df_filtered) / len(df) * 100:.1f}%) según filtros."
            )

        @st.cache_data
        def convert_df_to_csv(dataframe):
            return dataframe.to_csv(index=False).encode("utf-8")

        csv = convert_df_to_csv(df_filtered)
        st.download_button(
            label="📥 Descargar datos filtrados (CSV)",
            data=csv,
            file_name=f"neuro_support_analytics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Descarga los datos filtrados en formato CSV.",
        )

        st.markdown("---")

        df = df_filtered

        if df.empty:
            st.warning(
                "⚠️ No hay datos que coincidan con los filtros seleccionados."
            )
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de tickets", len(df))

        with col2:
            avg_churn = df["churn_risk_pred"].mean()
            st.metric("Churn promedio", f"{avg_churn:.1f}%")

        with col3:
            high_risk_pct = (
                (df["risk_segment"] == "Alto").sum() / len(df) * 100
            )
            st.metric("Tickets de alto riesgo", f"{high_risk_pct:.1f}%")

        with col4:
            phishing_count = df["is_phishing"].sum()
            st.metric("Tickets con phishing", phishing_count)

        st.markdown("---")

        overview_tab, churn_tab, sentiment_tab = st.tabs(
            ["Visión general", "Churn y riesgo", "Sentimiento y clientes"]
        )

        # Paletas distintas
        palette_types = ["#6366f1", "#22c55e", "#f97316", "#0ea5e9"]
        palette_risk = ["#f97316", "#e11d48", "#22c55e"]
        palette_hist = ["#0ea5e9", "#22c55e"]
        palette_channel = "Purples"
        palette_seg = "Magma"
        palette_sent_type = "Viridis"
        palette_sent_client = "Spectral"

        with overview_tab:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 📋 Distribución de tipos de ticket")
                type_counts = df["ticket_type_pred"].value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    hole=0.4,
                    color=type_counts.index,
                    color_discrete_sequence=palette_types,
                )
                fig.update_traces(
                    textposition="inside", textinfo="percent+label"
                )
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("### 📉 Segmentos de riesgo de churn")
                risk_counts = df["risk_segment"].value_counts()
                fig = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    color=risk_counts.index,
                    color_discrete_sequence=palette_risk,
                    labels={"x": "Segmento", "y": "Cantidad"},
                )
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)

        with churn_tab:
            st.markdown("### 📈 Distribución de riesgo de churn")
            fig = px.histogram(
                df,
                x="churn_risk_pred",
                nbins=20,
                color="ticket_type_pred",
                marginal="box",
                labels={
                    "churn_risk_pred": "Riesgo de churn (%)",
                    "ticket_type_pred": "Tipo de ticket",
                },
                color_discrete_sequence=palette_hist,
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

            col_c, col_d = st.columns(2)

            with col_c:
                st.markdown("### 📱 Churn promedio por canal")
                df["channel"] = df["channel"].replace(
                    {"dashboard": "Plataforma"}
                )
                channel_churn = (
                    df.groupby("channel")["churn_risk_pred"]
                    .mean()
                    .sort_values(ascending=False)
                )
                fig = px.bar(
                    x=channel_churn.index,
                    y=channel_churn.values,
                    labels={"x": "Canal", "y": "Churn promedio (%)"},
                    color=channel_churn.values,
                    color_continuous_scale=palette_channel,
                )
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)

            with col_d:
                st.markdown("### 🎯 Churn promedio por segmento")
                seg_churn = (
                    df.groupby("risk_segment")["churn_risk_pred"]
                    .mean()
                    .sort_values(ascending=False)
                )
                fig = px.bar(
                    x=seg_churn.index,
                    y=seg_churn.values,
                    labels={"x": "Segmento", "y": "Churn promedio (%)"},
                    color=seg_churn.values,
                    color_continuous_scale=palette_seg,
                )
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)

        with sentiment_tab:
            col_e, col_f = st.columns(2)

            with col_e:
                st.markdown(
                    "### 💬 Sentimiento promedio por tipo de ticket"
                )
                sentiment_by_type = df.groupby("ticket_type_pred")[
                    "sentiment_score"
                ].mean()
                fig = px.bar(
                    x=sentiment_by_type.index,
                    y=sentiment_by_type.values,
                    labels={
                        "x": "Tipo de ticket",
                        "y": "Sentimiento promedio",
                    },
                    color=sentiment_by_type.values,
                    color_continuous_scale=palette_sent_type,
                )
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)

            with col_f:
                st.markdown(
                    "### 😊 Sentimiento promedio por cliente (top 10)"
                )
                sentiment_by_client = (
                    df.groupby("client_name")["sentiment_score"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                fig = px.bar(
                    x=sentiment_by_client.index,
                    y=sentiment_by_client.values,
                    labels={"x": "Cliente", "y": "Sentimiento promedio"},
                    color=sentiment_by_client.values,
                    color_continuous_scale=palette_sent_client,
                )
                fig.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")


# ===========================
# VISTA 3: FACTORES DE CHURN
# ===========================
def view_churn_factors():
    """Vista 3: Análisis de factores de churn."""
    render_main_header()
    st.markdown(
        '<div class="view-header">🎯 Factores de Influencia en Churn</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    **Análisis de las variables que más impactan el riesgo de cancelación del cliente.**

    Esta información te ayuda a identificar señales tempranas de churn y tomar acción preventiva.
    """
    )

    try:
        df = load_gold_data()

        if df.empty:
            st.info(
                "ℹ️ No hay datos históricos aún. Procesa algunos tickets primero."
            )
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tickets analizados", len(df))

        with col2:
            avg_churn = df["churn_risk_pred"].mean()
            st.metric("Churn promedio", f"{avg_churn:.1f}%")

        with col3:
            high_risk_pct = (
                (df["risk_segment"] == "Alto").sum() / len(df) * 100
            )
            st.metric("Tickets de alto riesgo", f"{high_risk_pct:.1f}%")

        st.markdown("### 📊 Importancia de variables en predicción de churn")

        importance_data = {
            "Variable": [
                "Sentimiento del Cliente",
                "Incidentes Abiertos (30d)",
                "Antigüedad del Proyecto",
                "Longitud del Ticket",
                "Detección de Phishing",
            ],
            "Importancia": [0.467, 0.266, 0.159, 0.092, 0.017],
            "Descripción": [
                "El sentimiento negativo es el factor #1 de churn.",
                "Muchos tickets abiertos indican frustración acumulada.",
                "Proyectos más antiguos tienen mayor riesgo de desgaste.",
                "Tickets muy largos suelen indicar problemas complejos.",
                "Phishing es grave pero relativamente raro.",
            ],
        }

        importance_df = pd.DataFrame(importance_data)

        fig = px.bar(
            importance_df,
            x="Importancia",
            y="Variable",
            orientation="h",
            color="Importancia",
            color_continuous_scale="Blues",
            text="Importancia",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(
            showlegend=False, height=400, xaxis_title="Importancia relativa"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📝 Interpretación de factores")

        for _, row in importance_df.iterrows():
            with st.expander(
                f"**{row['Variable']}** — Importancia: {row['Importancia']:.1%}"
            ):
                st.write(row["Descripción"])

        st.markdown("---")
        st.markdown("### 🔗 Análisis de correlaciones")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Churn vs sentimiento")

            fig = px.scatter(
                df,
                x="sentiment_score",
                y="churn_risk_pred",
                color="ticket_type_pred",
                trendline="ols",
                labels={
                    "sentiment_score": "Sentimiento",
                    "churn_risk_pred": "Riesgo de churn (%)",
                    "ticket_type_pred": "Tipo de ticket",
                },
                color_discrete_sequence=["#6366f1", "#0ea5e9"],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "📌 **Insight:** Un sentimiento más negativo tiende a asociarse con mayor riesgo de churn."
            )

        with col2:
            st.markdown("#### Distribución de churn por tipo de ticket")

            fig = px.box(
                df,
                x="ticket_type_pred",
                y="churn_risk_pred",
                color="ticket_type_pred",
                labels={
                    "ticket_type_pred": "Tipo de ticket",
                    "churn_risk_pred": "Riesgo de churn (%)",
                },
                color_discrete_sequence=["#f97316", "#22c55e"],
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "📌 **Insight:** Los tickets correctivos suelen concentrar un riesgo de churn más alto."
            )

        st.markdown("---")
        st.markdown("### 💡 Recomendaciones para Account Managers")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Priorización")
            st.markdown(
                """
            - Enfócate en tickets con **sentimiento negativo**.  
            - Monitorea clientes con **3+ incidentes abiertos** en 30 días.  
            - Proyectos de **1+ año** requieren atención especial.  
            """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🚨 Señales de alerta")
            st.markdown(
                """
            - Cliente usa palabras como *frustrado*, *molesto*, *decepcionado*.  
            - Incremento súbito en tickets correctivos.  
            - Tickets largos con muchos detalles del problema.  
            """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### ✅ Acciones preventivas")
            st.markdown(
                """
            - Llamada proactiva antes de que el cliente escale.  
            - Revisar SLA y tiempos de respuesta recientes.  
            - Agendar una reunión de revisión trimestral.  
            """
            )
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error al analizar factores: {str(e)}")


# ===========================
# MAIN
# ===========================
def main():
    """Función principal del dashboard."""

    with st.sidebar:
        # LOGO REDONDO EN LA SIDEBAR (sin use_column_width)
        logo_path = Path("assets/logo.jpeg")
        if logo_path.exists():
            st.image(str(logo_path), width=160)

        st.markdown(
            "### 🧠 <span style='color: var(--ns-primary);'>Neuro Support</span>",
            unsafe_allow_html=True,
        )
        st.markdown("Plataforma de soporte inteligente con análisis de IA.")
        st.markdown("---")

        # Tema
        tema_label = st.radio(
            "Tema:",
            ["Oscuro", "Claro"],
            index=0 if st.session_state["tema"] == "oscuro" else 1,
        )
        st.session_state["tema"] = "oscuro" if tema_label == "Oscuro" else "claro"

        # Navegación
        view_option = st.radio(
            "Navegación:",
            ["Análisis de Ticket", "Análisis de Datos", "Factores de Churn"],
        )

        st.markdown("---")
        st.info(
            """
        **Neuro Support AI** te ayuda a:  
        - Detectar phishing y PII  
        - Clasificar tickets automáticamente  
        - Medir sentimiento del cliente  
        - Predecir riesgo de churn  
        """
        )

    pipeline = load_pipeline()

    if "Análisis de Ticket" in view_option:
        view_ticket_analyzer(pipeline)
    elif "Análisis de Datos" in view_option:
        view_data_analytics()
    elif "Factores de Churn" in view_option:
        view_churn_factors()

    st.markdown(
        """
    <div class="footer">
        NeuroSupport © 2025 — Desarrollado por Black_Cyber · Todos los derechos reservados
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
