"""
Módulo de Modelos de Machine Learning para clasificación y predicción.

Implementa modelos para:
- Clasificación de tipo de ticket (Correctivo/Evolutivo)
- Predicción de riesgo de churn

Owner: Data Science Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, mean_absolute_error, r2_score


class BaseModel:
    """
    Clase base para modelos de ML.
    
    Implementa el principio de Responsabilidad Única (SRP).
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa el modelo.
        
        Args:
            model_path: Ruta donde guardar/cargar el modelo
        """
        self.model = None
        self.model_path = model_path
        self.is_trained = False
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            path: Ruta donde guardar (opcional, usa self.model_path por defecto)
        """
        save_path = path or self.model_path
        if save_path and self.model:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"✅ Modelo guardado en {save_path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """
        Carga un modelo desde disco.
        
        Args:
            path: Ruta del modelo a cargar
        """
        load_path = path or self.model_path
        if load_path and load_path.exists():
            self.model = joblib.load(load_path)
            self.is_trained = True
            print(f"✅ Modelo cargado desde {load_path}")
        else:
            raise FileNotFoundError(f"No se encontró el modelo en {load_path}")


class TicketTypeClassifier(BaseModel):
    """
    Clasificador de tipo de ticket (Correctivo vs Evolutivo).
    
    Usa TF-IDF + Random Forest.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa el clasificador.
        
        Args:
            model_path: Ruta donde guardar/cargar el modelo
        """
        super().__init__(model_path)
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    
    def train(self, texts: pd.Series, labels: pd.Series) -> dict:
        """
        Entrena el clasificador de tipo de ticket.
        
        Args:
            texts: Serie con textos limpios de tickets
            labels: Serie con etiquetas ('Correctivo' o 'Evolutivo')
            
        Returns:
            dict: Métricas de evaluación
        """
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorizar textos
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_vec, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test_vec)
        
        self.is_trained = True
        
        return {
            'accuracy': self.model.score(X_test_vec, y_test),
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def predict(self, text: str) -> str:
        """
        Predice el tipo de ticket.
        
        Args:
            text: Texto limpio del ticket
            
        Returns:
            str: 'Correctivo' o 'Evolutivo'
        """
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado. Usa train() o load() primero.")
        
        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)[0]
        
        return prediction
    
    def save(self, path: Optional[Path] = None) -> None:
        """Guarda modelo y vectorizador."""
        save_path = path or self.model_path
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, save_path)
            print(f"✅ Clasificador guardado en {save_path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """Carga modelo y vectorizador."""
        load_path = path or self.model_path
        if load_path and load_path.exists():
            data = joblib.load(load_path)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.is_trained = True
            print(f"✅ Clasificador cargado desde {load_path}")
        else:
            raise FileNotFoundError(f"No se encontró el modelo en {load_path}")


class ChurnPredictor(BaseModel):
    """
    Predictor de riesgo de churn.
    
    Usa características numéricas + Random Forest Regressor.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Ruta donde guardar/cargar el modelo
        """
        super().__init__(model_path)
        self.feature_names = None
    
    def train(self, features: pd.DataFrame, churn_risk: pd.Series) -> dict:
        """
        Entrena el predictor de churn.
        
        Args:
            features: DataFrame con características numéricas:
                - project_age_days
                - open_incidents_30d
                - sentiment_label
                - is_phishing
                - word_count
            churn_risk: Serie con valores de churn (0-100)
            
        Returns:
            dict: Métricas de evaluación
        """
        self.feature_names = features.columns.tolist()
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, churn_risk, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test)
        
        self.is_trained = True
        
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict(self, features: dict) -> float:
        """
        Predice el riesgo de churn.
        
        Args:
            features: Diccionario con características:
                - project_age_days
                - open_incidents_30d
                - sentiment_score
                - is_phishing (0/1)
                - word_count
                
        Returns:
            float: Riesgo de churn (0-100)
        """
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado. Usa train() o load() primero.")
        
        # Crear DataFrame con las características en el orden correcto
        df = pd.DataFrame([features])[self.feature_names]
        
        # Predecir
        prediction = self.model.predict(df)[0]
        
        # Asegurar que esté en el rango [0, 100]
        return max(0, min(100, prediction))
    
    def get_feature_importance(self) -> dict:
        """
        Obtiene la importancia de cada característica.
        
        Returns:
            dict: {feature_name: importance_score}
        """
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado.")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Guarda modelo y feature_names.
        
        Args:
            path: Ruta donde guardar (opcional, usa self.model_path por defecto)
        """
        save_path = path or self.model_path
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, save_path)
            print(f"✅ Predictor guardado en {save_path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """
        Carga modelo y feature_names.
        
        Args:
            path: Ruta del modelo a cargar
        """
        load_path = path or self.model_path
        if load_path and load_path.exists():
            data = joblib.load(load_path)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.is_trained = True
            print(f"✅ Modelo cargado desde {load_path}")
        else:
            raise FileNotFoundError(f"No se encontró el modelo en {load_path}")


class ModelTrainer:
    """
    Orquestador para entrenar todos los modelos.
    
    Implementa el principio de Responsabilidad Única (SRP).
    """
    
    def __init__(self, data_path: Path, models_dir: Path):
        """
        Inicializa el entrenador.
        
        Args:
            data_path: Ruta al CSV de entrenamiento
            models_dir: Directorio donde guardar los modelos
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def train_all(self) -> dict:
        """
        Entrena todos los modelos y los guarda.
        
        Returns:
            dict: Métricas de todos los modelos
        """
        print("📊 Cargando datos de entrenamiento...")
        df = pd.read_csv(self.data_path)
        
        print(f"✅ Datos cargados: {len(df)} registros\n")
        
        # 1. Entrenar clasificador de tipo de ticket
        print("🤖 Entrenando clasificador de tipo de ticket...")
        ticket_classifier = TicketTypeClassifier(
            model_path=self.models_dir / "ticket_classifier.pkl"
        )
        classifier_metrics = ticket_classifier.train(df['text'], df['ticket_type'])
        ticket_classifier.save()
        
        print(f"   Accuracy: {classifier_metrics['accuracy']:.3f}")
        print(f"   Report:\n{classifier_metrics['classification_report']}\n")
        
        # 2. Entrenar predictor de churn
        print("🤖 Entrenando predictor de churn...")
        
        # Preparar características para churn
        churn_features = df[[
            'project_age_days',
            'open_incidents_30d',
            'sentiment_label',
            'is_phishing',
        ]].copy()
        
        # Agregar word_count simple
        churn_features['word_count'] = df['text'].str.split().str.len()
        
        churn_predictor = ChurnPredictor(
            model_path=self.models_dir / "churn_predictor.pkl"
        )
        churn_metrics = churn_predictor.train(churn_features, df['churn_risk'])
        churn_predictor.save()
        
        print(f"   MAE: {churn_metrics['mae']:.2f}")
        print(f"   R²: {churn_metrics['r2_score']:.3f}")
        print(f"   Feature Importance:")
        for feat, imp in sorted(churn_metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True):
            print(f"      {feat}: {imp:.3f}")
        
        return {
            'classifier': classifier_metrics,
            'churn_predictor': churn_metrics
        }


# Factory para crear entrenador
def create_model_trainer() -> ModelTrainer:
    """
    Crea una instancia de ModelTrainer con rutas por defecto.
    
    Returns:
        ModelTrainer: Entrenador configurado
    """
    data_path = Path("data/tickets_train.csv")
    models_dir = Path("models")
    
    return ModelTrainer(data_path, models_dir)
