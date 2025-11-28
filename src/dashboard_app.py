"""
Dashboard de Soporte Inteligente - Vortex AI

Dashboard interactivo en Streamlit con 3 vistas principales:
1. Análisis de ticket individual (Usuario de negocio)
2. Gráficos y distribuciones (Equipo de datos)
3. Análisis de factores de churn (Account Manager)

Ejecutar con: python3 -m streamlit run src/dashboard_app.py

Owner: Product Team
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import base64

# Agregar src al path para imports
sys.path.insert(0, str(Path(__file__).parent))

from nlp_pipeline import create_pipeline
from db_utils import get_connection


# Configuración de la página
st.set_page_config(
    page_title="Neuro Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados - Tema Neuro Support (Dark Blue/Tech)
st.markdown("""
<style>

/* ===========================
   FUENTES
=========================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===========================
   BACKGROUND GENERAL
=========================== */
.stApp {
    background: radial-gradient(circle at top left, #00111a, #000c12 40%, #00060c 90%);
    color: #D7E2EB;
}

/* ===========================
   SIDEBAR
=========================== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000B13 0%, #00131E 100%);
    border-right: 1px solid rgba(0, 180, 216, 0.12);
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #00B4D8 !important;
}

.sidebar-logo {
    margin-bottom: 18px;
    padding: 10px;
    background: rgba(0, 180, 216, 0.08);
    border-radius: 12px;
    border: 1px solid rgba(0,180,216,0.2);
}

/* ===========================
   HEADER NEURO SUPPORT
=========================== */
.neuro-header {
    font-family: 'Inter', sans-serif;
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 22px 28px;
    margin-bottom: 2.5rem;

    background: rgba(0, 119, 182, 0.10);
    border: 1px solid rgba(0, 180, 216, 0.25);
    box-shadow: 0 0 25px rgba(0, 180, 216, 0.20);
    backdrop-filter: blur(14px);
    border-radius: 16px;
}

.neuro-logo {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #00B4D8;
    box-shadow: 0 0 15px rgba(0, 180, 216, 0.5);
    animation: pulse 2s infinite;
}

.neuro-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #90E0EF, #00B4D8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.neuro-badge {
    background: rgba(0, 180, 216, 0.15);
    border: 1px solid rgba(0, 180, 216, 0.35);
    color: #90E0EF;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.neuro-sub {
    margin-top: -8px;
    font-size: 0.95rem;
    color: #A8D8E9;
}

/* ===========================
   HEADERS INTERNOS
=========================== */
.view-header {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1.4rem;
    background: linear-gradient(90deg, #48CAE4, #00B4D8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 14px rgba(0,180,216,0.25);
}

/* ===========================
   CARDS
=========================== */
.metric-card {
    background: rgba(255,255,255,0.02);
    padding: 1.4rem;
    border-radius: 14px;
    border: 1px solid rgba(0, 180, 216, 0.15);
    backdrop-filter: blur(6px);
    transition: transform 0.25s ease, border-color 0.3s;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,180,216,0.4);
}

/* ===========================
   BOTONES
=========================== */
.stButton > button {
    background: linear-gradient(90deg, #0077B6, #0096C7);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: bold;
    transition: 0.3s ease;
    padding: 0.6rem 1.2rem;
}

.stButton > button:hover {
    box-shadow: 0 0 18px rgba(0, 180, 216, 0.45);
    transform: scale(1.02);
}

/* ===========================
   INPUTS
=========================== */
.stTextInput > div > input,
.stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    color: #D7E2EB !important;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #D7E2EB !important;
    border-radius: 8px !important;
}

/* ===========================
   ALERTAS
=========================== */
.warning-box {
    background-color: rgba(255, 193, 7, 0.08);
    border-left: 4px solid #FFC107;
    padding: 1rem;
    border-radius: 8px;
}

.danger-box {
    background-color: rgba(220, 53, 69, 0.1);
    border-left: 4px solid #DC3545;
    padding: 1rem;
    border-radius: 8px;
}

.success-box {
    background-color: rgba(25, 135, 84, 0.08);
    border-left: 4px solid #198754;
    padding: 1rem;
    border-radius: 8px;
}

/* ===========================
   FOOTER
=========================== */
.footer {
    margin-top: 2rem;
    text-align: center;
    font-size: 0.8rem;
    opacity: 0.6;
}

/* ===========================
   TEXTO Y VISIBILIDAD
=========================== */
/* Labels de todos los widgets (Inputs, Selects, Sliders) */
.stTextArea label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stDateInput label, .stMultiSelect label {
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 1rem;
}

/* Texto general en Markdown y Sidebar */
.stMarkdown p, .stSidebar .stMarkdown p {
    color: #D7E2EB !important;
}

/* Títulos de Métricas */
[data-testid="stMetricLabel"] {
    color: #A0AEC0 !important;
    font-size: 0.9rem;
}

/* Valores de Métricas */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Texto de ayuda (pequeño) */
.stNumberInput div[data-testid="stMarkdownContainer"] p {
    color: #A0AEC0 !important;
}

/* Radio buttons en Sidebar */
.stRadio label {
    color: #FFFFFF !important;
}

/* Expander headers */
.streamlit-expanderHeader {
    color: #FFFFFF !important;
    background-color: rgba(255,255,255,0.05) !important;
}

/* Botón de cerrar/abrir Sidebar y Menú */
[data-testid="stSidebarCollapsedControl"] {
    color: #FFFFFF !important;
}

[data-testid="stSidebarCollapsedControl"] svg, 
[data-testid="stHeader"] svg {
    fill: #FFFFFF !important;
}

/* Ocultar decoración superior de colores de Streamlit */
[data-testid="stDecoration"] {
    display: none;
}

@keyframes pulse {
    0% { transform: scale(1); text-shadow: 0 0 0 rgba(0, 180, 216, 0.7); }
    50% { transform: scale(1.05); text-shadow: 0 0 20px rgba(0, 180, 216, 0.7); }
    100% { transform: scale(1); text-shadow: 0 0 0 rgba(0, 180, 216, 0.7); }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """
    Carga el pipeline de NLP con modelos entrenados.
    
    Usa @st.cache_resource para cargar una sola vez.
    """
    # Obtener path absoluto basado en la ubicación de este archivo
    current_file = Path(__file__).parent
    models_dir = current_file.parent / "models"
    
    if not models_dir.exists() or not (models_dir / "ticket_classifier.pkl").exists():
        st.error(f"⚠️ Los modelos no están entrenados. Ruta buscada: {models_dir}")
        st.error("Ejecuta primero: python3 src/train_models.py")
        st.stop()
    
    return create_pipeline(models_dir)


@st.cache_data
def load_gold_data():
    """
    Carga datos de la tabla GOLD para análisis.
    
    Returns:
        pd.DataFrame: Datos de predicciones históricas
    """
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
    except Exception as e:
        # Si hay error, retornar DataFrame vacío
        return pd.DataFrame()


def view_ticket_analyzer(pipeline):
    """
    Vista 1: Formulario para analizar tickets individuales.
    
    Dirigida a usuarios de negocio que quieren analizar un ticket nuevo.
    """

    # Cargar logo para el header
    logo_path = Path("assets/logo.jpeg")
    logo_html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{encoded}" class="neuro-logo">'
    else:
        logo_html = '<div class="neuro-logo">🧠</div>'

    # Nuevo Header profesional
    st.markdown(f"""
    <div class="neuro-header">
        {logo_html}
        <div>
            <div class="neuro-title">Neuro Support AI</div>
            <div class="neuro-sub">Análisis inteligente de tickets, seguridad y predicción de churn</div>
        </div>
        <div class="neuro-badge">BETA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Pega el texto del ticket** y obtén un análisis completo con IA:
    - 🔒 Detección de seguridad (phishing, PII)
    - 📊 Clasificación del tipo de ticket
    - 📉 Predicción de riesgo de churn
    - 💡 Recomendaciones de acción
    """)

    # Formulario de entrada
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticket_text = st.text_area(
            "Texto del Ticket:",
            height=200,
            placeholder="Ejemplo: El módulo de facturación está lanzando error 500. El cliente está muy molesto y quiere una solución urgente..."
        )
    
    with col2:
        st.markdown("**Información del Proyecto:**")
        project_age = st.number_input(
            "Antigüedad del proyecto (días):",
            min_value=1,
            max_value=3650,
            value=180,
            help="¿Hace cuántos días se inició el proyecto?"
        )
        
        open_incidents = st.number_input(
            "Incidentes abiertos (últimos 30d):",
            min_value=0,
            max_value=100,
            value=2,
            help="Número de tickets abiertos en los últimos 30 días"
        )
    
    # Botón de análisis
    if st.button("🔍 Analizar Ticket", type="primary", use_container_width=True):
        if not ticket_text.strip():
            st.warning("⚠️ Por favor ingresa el texto del ticket")
            return
        
        with st.spinner("Analizando ticket..."):
            try:
                # Procesar ticket
                result = pipeline.process(
                    text=ticket_text,
                    project_age_days=int(project_age),
                    open_incidents_30d=int(open_incidents)
                )
                
                # Mostrar resultados
                st.markdown("---")
                st.markdown("### 📋 Resultados del Análisis")
                
                # Alertas de seguridad
                if result.is_phishing or result.has_pii:
                    st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                    st.markdown("#### 🚨 ALERTAS DE SEGURIDAD")
                    if result.is_phishing:
                        st.error("⚠️ **PHISHING DETECTADO**: Este ticket contiene patrones sospechosos de phishing")
                    if result.has_pii:
                        st.warning("⚠️ **PII DETECTADO**: El ticket contiene información personal identificable")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Tipo de Ticket",
                        result.ticket_type_pred,
                        delta="Correctivo" if result.ticket_type_pred == "Correctivo" else "Evolutivo",
                        delta_color="inverse" if result.ticket_type_pred == "Correctivo" else "normal"
                    )
                
                with col2:
                    risk_color = "🔴" if result.risk_segment == "Alto" else ("🟡" if result.risk_segment == "Medio" else "🟢")
                    st.metric(
                        "Riesgo de Churn",
                        f"{result.churn_risk_pred:.1f}%",
                        delta=f"{risk_color} {result.risk_segment}"
                    )
                
                with col3:
                    sentiment_emoji = "😠" if result.sentiment_score < -0.3 else ("😐" if result.sentiment_score < 0.3 else "😊")
                    st.metric(
                        "Sentimiento",
                        f"{result.sentiment_score:.2f}",
                        delta=f"{sentiment_emoji}"
                    )
                
                with col4:
                    st.metric(
                        "Palabras",
                        result.word_count
                    )
                
                # Recomendaciones
                st.markdown("---")
                st.markdown("### 💡 Recomendaciones")
                
                # Color según riesgo
                if result.risk_segment == "Alto":
                    st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                elif result.risk_segment == "Medio":
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                
                st.markdown(result.recommendation_text.replace('\n', '\n\n'))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Texto procesado (expandible)
                with st.expander("📝 Ver texto procesado"):
                    st.text_area("Texto limpio:", result.cleaned_text, height=150, disabled=True)
                
            except Exception as e:
                import traceback
                st.error(f"❌ Error al procesar el ticket: {type(e).__name__}: {str(e)}")
                st.code(traceback.format_exc())



def view_data_analytics():
    """
    Vista 2: Gráficos y análisis de datos.
    
    Dirigida al equipo de datos para explorar distribuciones y patrones.
    """
    st.markdown('<div class="view-header">📊 Análisis de Datos</div>', unsafe_allow_html=True)
    
    try:
        df = load_gold_data()
        
        if df.empty:
            st.info("ℹ️ No hay datos históricos aún. Procesa algunos tickets primero.")
            return
        
        # Convertir created_at a datetime si es string
        if df['created_at'].dtype == 'object':
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # === FILTROS ===
        st.markdown("### 🔍 Filtros")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Filtro de fechas
            min_date = df['created_at'].min().date()
            max_date = df['created_at'].max().date()
            
            date_range = st.date_input(
                "Rango de fechas:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filtra tickets por fecha de creación"
            )
        
        with col2:
            # Filtro de clientes
            all_clients = sorted(df['client_name'].unique().tolist())
            selected_clients = st.multiselect(
                "Clientes:",
                options=all_clients,
                default=all_clients,
                help="Selecciona uno o más clientes"
            )
        
        with col3:
            # Filtro de canales
            all_channels = sorted(df['channel'].unique().tolist())
            selected_channels = st.multiselect(
                "Canales:",
                options=all_channels,
                default=all_channels,
                help="Filtra por canal de comunicación"
            )
        
        with col4:
            # Filtro de segmento de riesgo
            all_segments = ['Bajo', 'Medio', 'Alto']
            selected_segments = st.multiselect(
                "Segmento de Riesgo:",
                options=all_segments,
                default=all_segments,
                help="Filtra por nivel de riesgo de churn"
            )
        
        # Aplicar filtros
        df_filtered = df.copy()
        
        # Filtro de fechas
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_filtered[
                (df_filtered['created_at'].dt.date >= start_date) &
                (df_filtered['created_at'].dt.date <= end_date)
            ]
        
        # Filtro de clientes
        if selected_clients:
            df_filtered = df_filtered[df_filtered['client_name'].isin(selected_clients)]
        
        # Filtro de canales
        if selected_channels:
            df_filtered = df_filtered[df_filtered['channel'].isin(selected_channels)]
        
        # Filtro de segmentos
        if selected_segments:
            df_filtered = df_filtered[df_filtered['risk_segment'].isin(selected_segments)]
        
        # Mostrar info de filtros aplicados
        if len(df_filtered) < len(df):
            st.info(f"📊 Mostrando {len(df_filtered)} de {len(df)} tickets ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # Botón para exportar datos filtrados
        @st.cache_data
        def convert_df_to_csv(dataframe):
            return dataframe.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(df_filtered)
        st.download_button(
            label="📥 Descargar datos filtrados (CSV)",
            data=csv,
            file_name=f"vortex_analytics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Descarga los datos filtrados en formato CSV"
        )
        
        st.markdown("---")
        
        # Usar df_filtered en lugar de df para el resto
        df = df_filtered
        
        if df.empty:
            st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
            return
        
        # Métricas resumen
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickets", len(df))
        
        with col2:
            avg_churn = df['churn_risk_pred'].mean()
            st.metric("Churn Promedio", f"{avg_churn:.1f}%")
        
        with col3:
            high_risk_pct = (df['risk_segment'] == 'Alto').sum() / len(df) * 100
            st.metric("Tickets Alto Riesgo", f"{high_risk_pct:.1f}%")
        
        with col4:
            phishing_count = df['is_phishing'].sum()
            st.metric("Phishing Detectados", phishing_count)
        
        st.markdown("---")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de tipos de ticket
            st.markdown("### 📋 Distribución de Tipos de Ticket")
            type_counts = df['ticket_type_pred'].value_counts()
            
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                color_discrete_sequence=['#ff7f0e', '#1f77b4'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribución de segmentos de riesgo
            st.markdown("### 📉 Segmentos de Riesgo de Churn")
            risk_counts = df['risk_segment'].value_counts()
            
            colors = {'Alto': '#dc3545', 'Medio': '#ffc107', 'Bajo': '#28a745'}
            
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                color=risk_counts.index,
                color_discrete_map=colors,
                labels={'x': 'Segmento', 'y': 'Cantidad'}
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribución de churn risk
        st.markdown("### 📈 Distribución de Riesgo de Churn")
        
        fig = px.histogram(
            df,
            x='churn_risk_pred',
            nbins=20,
            color='ticket_type_pred',
            marginal='box',
            labels={'churn_risk_pred': 'Riesgo de Churn (%)', 'ticket_type_pred': 'Tipo'},
            color_discrete_sequence=['#ff7f0e', '#1f77b4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Churn por canal
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📱 Churn Promedio por Canal")
            channel_churn = df.groupby('channel')['churn_risk_pred'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=channel_churn.index,
                y=channel_churn.values,
                labels={'x': 'Canal', 'y': 'Churn Promedio (%)'},
                color=channel_churn.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💬 Sentimiento Promedio por Tipo")
            sentiment_by_type = df.groupby('ticket_type_pred')['sentiment_score'].mean()
            
            fig = px.bar(
                x=sentiment_by_type.index,
                y=sentiment_by_type.values,
                labels={'x': 'Tipo de Ticket', 'y': 'Sentimiento Promedio'},
                color=sentiment_by_type.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")


def view_churn_factors():
    """
    Vista 3: Análisis de factores de churn.
    
    Dirigida a Account Managers para entender qué variables influyen más.
    """
    st.markdown('<div class="view-header">🎯 Factores de Influencia en Churn</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Análisis de las variables que más impactan el riesgo de cancelación del cliente.**
    
    Esta información te ayuda a identificar señales tempranas de churn y tomar acción preventiva.
    """)
    
    try:
        df = load_gold_data()
        
        if df.empty:
            st.info("ℹ️ No hay datos históricos aún. Procesa algunos tickets primero.")
            return
        
        # Importancia de características (hardcoded del modelo)
        st.markdown("### 📊 Importancia de Variables en Predicción de Churn")
        
        # Estos valores vienen del entrenamiento
        importance_data = {
            'Variable': [
                'Sentimiento del Cliente',
                'Incidentes Abiertos (30d)',
                'Antigüedad del Proyecto',
                'Longitud del Ticket',
                'Detección de Phishing'
            ],
            'Importancia': [0.467, 0.266, 0.159, 0.092, 0.017],
            'Descripción': [
                'El sentimiento negativo es el factor #1 de churn',
                'Muchos tickets abiertos indican frustración acumulada',
                'Proyectos más antiguos tienen mayor riesgo',
                'Tickets muy largos suelen indicar problemas complejos',
                'Phishing es grave pero relativamente raro'
            ]
        }
        
        importance_df = pd.DataFrame(importance_data)
        
        # Gráfico de importancia
        fig = px.bar(
            importance_df,
            x='Importancia',
            y='Variable',
            orientation='h',
            color='Importancia',
            color_continuous_scale='Blues',
            text='Importancia'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(showlegend=False, height=400, xaxis_title="Importancia Relativa")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla con descripciones
        st.markdown("### 📝 Interpretación de Factores")
        
        for _, row in importance_df.iterrows():
            with st.expander(f"**{row['Variable']}** - Importancia: {row['Importancia']:.1%}"):
                st.write(row['Descripción'])
        
        st.markdown("---")
        
        # Análisis de correlaciones
        st.markdown("### 🔗 Análisis de Correlaciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn vs Sentimiento
            st.markdown("#### Churn vs Sentimiento")
            
            fig = px.scatter(
                df,
                x='sentiment_score',
                y='churn_risk_pred',
                color='ticket_type_pred',
                trendline='ols',
                labels={
                    'sentiment_score': 'Sentimiento',
                    'churn_risk_pred': 'Riesgo de Churn (%)',
                    'ticket_type_pred': 'Tipo'
                },
                color_discrete_sequence=['#ff7f0e', '#1f77b4']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("📌 **Insight:** Sentimiento negativo correlaciona fuertemente con mayor churn")
        
        with col2:
            # Churn por tipo de ticket
            st.markdown("#### Distribución de Churn por Tipo")
            
            fig = px.box(
                df,
                x='ticket_type_pred',
                y='churn_risk_pred',
                color='ticket_type_pred',
                labels={
                    'ticket_type_pred': 'Tipo de Ticket',
                    'churn_risk_pred': 'Riesgo de Churn (%)'
                },
                color_discrete_sequence=['#ff7f0e', '#1f77b4']
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("📌 **Insight:** Tickets correctivos tienden a tener mayor riesgo de churn")
        
        # Recomendaciones para Account Managers
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones para Account Managers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Priorización")
            st.markdown("""
            - Enfócate en tickets con **sentimiento negativo**
            - Monitorea clientes con **3+ incidentes abiertos**
            - Proyectos de **1+ año** requieren atención especial
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🚨 Señales de Alerta")
            st.markdown("""
            - Cliente usa palabras como "frustrado", "molesto"
            - Incremento súbito en tickets correctivos
            - Tickets largos con muchos detalles del problema
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### ✅ Acciones Preventivas")
            st.markdown("""
            - Llamada proactiva antes de que escale
            - Revisar SLA y tiempos de respuesta
            - Considerar reunión de revisión trimestral
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error al analizar factores: {str(e)}")


def main():
    """Función principal del dashboard."""
    
    # Sidebar con navegación
    with st.sidebar:
        logo_path = Path("assets/logo.jpeg")
        if logo_path.exists():
            st.image(str(logo_path), width=180, caption="NeuroSupport by Black_Cyber")

        st.markdown("<h2 style='color:#00B4D8;'>🧠 Neuro Support</h2>", unsafe_allow_html=True)
        st.markdown("Plataforma de Soporte Inteligente.")
        st.markdown("---")

        view_option = st.radio(
            "Navegación:",
            ["Análisis de Ticket", "Análisis de Datos", "Factores de Churn"]
        )

        st.markdown("---")
        st.info("""
        **Neuro Support AI**:
        - Detección de phishing y PII  
        - Clasificación inteligente  
        - Predicción de churn  
        - Recomendaciones accionables  
        """)
    
    # Cargar pipeline
    pipeline = load_pipeline()
    
    # Renderizar vista seleccionada
    if "Análisis de Ticket" in view_option:
        view_ticket_analyzer(pipeline)
    elif "Análisis de Datos" in view_option:
        view_data_analytics()
    elif "Factores de Churn" in view_option:
        view_churn_factors()
    
    # Footer
    st.markdown("""
    <div class="footer">
        NeuroSupport © 2025 — Desarrollado por Black_Cyber · Todos los derechos reservados
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
