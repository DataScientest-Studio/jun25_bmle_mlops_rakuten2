"""
Schémas Pydantic pour l'API Rakuten MLOps.
==========================================

Ce module définit les modèles de données pour les requêtes et réponses de l'API.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field


# ============================================================================
# Schémas pour l'endpoint /predict
# ============================================================================
# Note: PredictionRequest supprimé - l'API utilise uniquement multipart/form-data
# Les paramètres sont passés directement via Form() dans main.py


class PredictionResponse(BaseModel):
    """Réponse de prédiction pour un produit."""
    prediction: int = Field(
        ...,
        description="Code type produit prédit",
    )
    confidence: Optional[float] = Field(
        None,
        description="Confiance de la prédiction (0-1)",
    )
    top_classes: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Top 3 des classes avec probabilités",
    )


# ============================================================================
# Schémas pour l'endpoint /training
# ============================================================================


class TrainingRequest(BaseModel):
    """Requête pour lancer un entraînement."""
    model_name: Optional[str] = Field(
        "lr",
        description="Type de modèle (lr, svc, xgb, lgbm)",
    )
    experiment_name: Optional[str] = Field(
        "rakuten_classification",
        description="Nom de l'expérience MLflow",
    )
    run_name: Optional[str] = Field(
        None,
        description="Nom du run MLflow (optionnel)",
    )
    dag_run_id: Optional[str] = Field(
        None,
        description="Identifiant du run Airflow (pour la promotion automatique dans MLflow)",
    )


class TrainingResponse(BaseModel):
    """Réponse après entraînement."""
    status: str = Field(
        ...,
        description="success ou error",
    )
    message: str = Field(
        ...,
        description="Message descriptif",
    )
    model_path: Optional[str] = Field(
        None,
        description="Chemin du modèle sauvegardé",
    )
    metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Métriques d'entraînement (valeurs numériques)",
    )
    mlflow_run_id: Optional[str] = Field(
        None,
        description="ID du run MLflow",
    )
    mlflow_experiment_id: Optional[str] = Field(
        None,
        description="ID de l'expérience MLflow",
    )
    training_time: Optional[float] = Field(
        None,
        description="Temps d'entraînement (secondes)",
    )


# ============================================================================
# Schémas généraux
# ============================================================================


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str = Field(
        ...,
        description="healthy ou degraded",
    )
    service: str = Field(
        ...,
        description="Nom du service",
    )
    version: str = Field(
        ...,
        description="Version de l'API",
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp du check",
    )
    database: Optional[str] = Field(
        None,
        description="État de la base de données",
    )
    model_loaded: Optional[bool] = Field(
        None,
        description="Modèle chargé ou non",
    )


class ModelInfoResponse(BaseModel):
    """Information sur le modèle chargé."""
    model_type: str = Field(
        ...,
        description="Type de modèle",
    )
    model_path: str = Field(
        ...,
        description="Chemin du modèle",
    )
    n_features: Optional[int] = Field(
        None,
        description="Nombre de features attendues",
    )
    n_classes: Optional[int] = Field(
        None,
        description="Nombre de classes",
    )
    classes: Optional[List[int]] = Field(
        None,
        description="Liste des classes",
    )
    has_label_encoder: bool = Field(
        ...,
        description="Présence d'un LabelEncoder",
    )
