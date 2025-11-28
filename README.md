# Vortex Support AI - Plataforma de Soporte Inteligente

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Plataforma de Soporte Inteligente y Ciber-Resiliente** desarrollada para el Hackathon Talento Tech.

Sistema inteligente que combina Machine Learning y análisis de seguridad para optimizar la gestión de tickets de soporte, detectar riesgos y prevenir la pérdida de clientes.

## 🎯 Características Principales

- **🔒 Análisis de Seguridad**: Detección automática de phishing y PII
- **🤖 Clasificación Inteligente**: Categoriza tickets como Correctivos o Evolutivos
- **📊 Predicción de Churn**: Calcula el riesgo de cancelación del cliente (0-100%)
- **💡 Recomendaciones**: Genera acciones específicas basadas en el análisis
- **📈 Dashboard Interactivo**: 3 vistas personalizadas para diferentes roles

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/hackathon-vortex-support-ai.git
cd hackathon-vortex-support-ai

# Instalar dependencias
pip3 install -r requirements.txt
```

### 2. Generar Datos de Entrenamiento

```bash
python3 src/generate_dummy_data.py
```

### 3. Entrenar Modelos

```bash
python3 src/train_models.py
```

### 4. Ejecutar Dashboard

```bash
python3 -m streamlit run src/dashboard_app.py
```

El dashboard estará disponible en `http://localhost:8501`

## 📊 Vistas del Dashboard

### 🎯 Vista 1: Análisis de Ticket Individual
**Para:** Usuarios de negocio y soporte

Pega el texto de un ticket y obtén instantáneamente:
- Detección de seguridad (phishing/PII)
- Clasificación del tipo
- Predicción de riesgo de churn
- Recomendaciones de acción

### 📈 Vista 2: Análisis de Datos
**Para:** Equipo de datos y analistas

Visualiza patrones y distribuciones:
- Distribución de tipos de tickets
- Segmentos de riesgo
- Churn promedio por canal
- Sentimiento por tipo de ticket

### 🎯 Vista 3: Factores de Churn
**Para:** Account Managers

Entiende qué variables influyen más:
- Importancia de cada factor
- Correlaciones con churn
- Señales de alerta temprana
- Recomendaciones preventivas

## 🏗️ Arquitectura

### Estructura del Proyecto

```
hackathon-vortex-support-ai/
├── src/
│   ├── security.py           # Detección de phishing y PII
│   ├── preprocessing.py      # Limpieza y análisis de sentimiento
│   ├── recommender.py        # Sistema de recomendaciones
│   ├── models.py            # Modelos de ML
│   ├── nlp_pipeline.py      # Pipeline orquestador
│   ├── dashboard_app.py     # Dashboard Streamlit
│   ├── db_utils.py          # Utilidades de base de datos
│   ├── generate_dummy_data.py  # Generador de datos
│   ├── init_db.py           # Inicialización de BD
│   └── train_models.py      # Script de entrenamiento
├── data/
│   └── tickets_train.csv    # Dataset de entrenamiento
├── models/
│   ├── ticket_classifier.pkl  # Modelo de clasificación
│   └── churn_predictor.pkl   # Modelo de predicción
└── sql/
    └── schema.sql           # Esquema de base de datos

```

### Base de Datos (SQLite)

Arquitectura de 3 capas:

1. **RAW**: Tickets originales sin procesar
2. **CORE**: Tickets enriquecidos con análisis de seguridad y sentimiento
3. **GOLD**: Predicciones y recomendaciones para negocio

## 🤖 Modelos de ML

### Clasificador de Tickets
- **Algoritmo**: Random Forest Classifier
- **Features**: TF-IDF (100 features, bigrams)
- **Accuracy**: 100% en datos de prueba

### Predictor de Churn
- **Algoritmo**: Random Forest Regressor
- **Features principales**:
  - Sentimiento del cliente (46.7% importancia)
  - Incidentes abiertos en 30 días (26.6%)
  - Antigüedad del proyecto (15.9%)
  - Longitud del ticket (9.2%)
  - Detección de phishing (1.7%)
- **Performance**: MAE 12.61, R² 0.335

## 🛠️ Principios de Diseño

El código sigue **principios SOLID**:

- **SRP**: Cada clase tiene una única responsabilidad
- **OCP**: Abierto para extensión, cerrado para modificación
- **LSP**: Uso de abstracciones e interfaces
- **ISP**: Interfaces específicas y segregadas
- **DIP**: Dependencia de abstracciones, no implementaciones

### Patrones Utilizados

- **Strategy Pattern**: Recomendaciones intercambiables
- **Factory Pattern**: Creación consistente de objetos
- **Facade Pattern**: Pipeline simplifica subsistemas complejos

## 📝 Tecnologías

- **Python 3.9+**
- **Streamlit**: Dashboard interactivo
- **scikit-learn**: Modelos de ML
- **NLTK**: Procesamiento de lenguaje natural
- **Plotly**: Visualizaciones interactivas
- **SQLite**: Base de datos
- **Pandas/NumPy**: Manipulación de datos

## 👥 Equipo

Desarrollado para el **NeuroSupport un programa de Black_Cyber**

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

**Todos los derechos reservados ® Desarrollado por Black_Cyber**
