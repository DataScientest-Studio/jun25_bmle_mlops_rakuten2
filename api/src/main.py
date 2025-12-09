"""
API FastAPI pour le projet Rakuten MLOps.
=========================================

Cette API expose deux endpoints principaux:
- POST /training/ : Lance l'entraînement d'un modèle
- POST /predict/ : Fait des prédictions sur de nouvelles données (non implémenté pour le moment)

Démarrage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
# In Docker: main.py is at /app/main.py, so parent is /app/
# Locally: main.py is at api/src/main.py, so parent.parent is api/
if Path(__file__).parent.name == "src":
    # Running locally: api/src/main.py -> root_dir = api/
    root_dir = Path(__file__).parent.parent
else:
    # Running in Docker: /app/main.py -> root_dir = /app/
    root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

import logging
import time
from datetime import datetime
import re  # pour nettoyer le nom de modèle

import psycopg2
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    HealthResponse,
    ModelInfoResponse,
)

# Imports MLflow / Joblib pour le tracking et le Model Registry
import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.models.predict_model import ProductPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Nom du modèle enregistré dans le Model Registry MLflow
# (doit être le même que dans la tâche Airflow de promotion)
REGISTERED_MODEL_NAME = "model"
# Alias utilisé par Airflow pour la version "Production"
PRODUCTION_ALIAS = "production"

# =======================================================================
# Configuration de l'application
# =======================================================================

app = FastAPI(
    title="Rakuten MLOps API",
    description="""
    API pour la classification de produits e-commerce Rakuten.
    
    ## Endpoints principaux
    
    * **POST /training/** - Lance l'entraînement d'un modèle ML
    * **POST /predict/** - Prédit la catégorie d'un produit (non implémenté)
    * **GET /health** - Vérification de l'état de l'API
    * **GET /model/info** - Informations sur le modèle chargé
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
predictor = None
DATABASE_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "localhost"),
    "port": int(os.getenv("DATABASE_PORT", "5433")),
    "database": os.getenv("DATABASE_NAME", "rakuten"),
    "user": os.getenv("DATABASE_USER", "mlops"),
    "password": os.getenv("DATABASE_PASSWORD", "mlops"),
}

# =======================================================================
# Événements
# =======================================================================


@app.on_event("startup")
async def startup_event():
    """Actions au démarrage."""
    logger.info("Démarrage de l'API Rakuten MLOps...")

    try:
        check_database_connection()
        logger.info("Connexion à PostgreSQL réussie")
    except Exception as e:
        logger.warning(f"Connexion à PostgreSQL échouée: {e}")

    # 1) Essayer de charger le modèle 'production' depuis MLflow
    loaded = load_production_model_from_mlflow()
    if not loaded:
        logger.warning(
            "Aucun modèle 'production' chargé depuis MLflow, tentative de chargement local."
        )
        try:
            load_default_model()
        except Exception as e:
            logger.warning(f"Aucun modèle local chargé: {e}")

    if predictor is None:
        logger.warning("Aucun modèle n'est chargé au démarrage de l'API.")
    else:
        logger.info("Modèle de prédiction prêt.")

    logger.info("API prête à recevoir des requêtes")


# =======================================================================
# Fonctions utilitaires
# =======================================================================


def check_database_connection() -> bool:
    """Vérifie la connexion à PostgreSQL."""
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM project.items;")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        logger.info(f"Base de données: {count} produits disponibles")
        return True
    except Exception as e:
        logger.error(f"Erreur de connexion à la base: {e}")
        raise


def load_default_model():
    """Charge le modèle par défaut si disponible (fallback local)."""
    global predictor

    models_dir = root_dir / "models"

    if not models_dir.exists():
        logger.warning("Dossier models/ non trouvé")
        return

    model_files = list(models_dir.glob("*_final.joblib"))

    if not model_files:
        model_files = list(models_dir.glob("*.joblib"))

    if not model_files:
        logger.warning("Aucun modèle .joblib trouvé")
        return

    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Chargement du modèle local: {latest_model}")

    predictor = ProductPredictor(str(latest_model))

    logger.info(f"Modèle chargé: {predictor.get_model_info()['model_type']}")


def load_production_model_from_mlflow() -> bool:
    """
    Tente de charger le modèle avec l'alias 'production' depuis le Model Registry MLflow.
    On va chercher l'artifact 'models/model_full_pipeline.joblib' en priorité,
    puis 'models/model.joblib' en fallback.
    Retourne True si succès, False sinon.
    """
    global predictor

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        # 1) Récupérer la version qui a l'alias 'production'
        mv = client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=PRODUCTION_ALIAS,
        )
    except Exception as e:
        logger.warning(
            "Impossible de récupérer le modèle '%s' avec l'alias '%s' dans MLflow: %s",
            REGISTERED_MODEL_NAME,
            PRODUCTION_ALIAS,
            e,
        )
        return False

    logger.info(
        "Version 'production' trouvée dans le registry: name=%s version=%s run_id=%s",
        mv.name,
        mv.version,
        mv.run_id,
    )

    # 2) Télécharger l'artifact du run correspondant
    try:
        local_models_dir = root_dir / "models" / "production"
        local_models_dir.mkdir(parents=True, exist_ok=True)

        artifact_candidates = [
            "models/model_full_pipeline.joblib",
            "models/model.joblib",
        ]

        model_path = None
        last_error = None

        for artifact_rel_path in artifact_candidates:
            try:
                logger.info(
                    "Tentative de téléchargement de l'artifact MLflow: %s",
                    artifact_rel_path,
                )
                local_path = client.download_artifacts(
                    run_id=mv.run_id,
                    path=artifact_rel_path,
                    dst_path=str(local_models_dir),
                )
                p = Path(local_path)
                if p.is_dir():
                    # cas défensif: si MLflow renvoie un dossier,
                    # on prend le fichier final du chemin
                    p = p / Path(artifact_rel_path).name

                if p.exists():
                    model_path = p
                    logger.info("Artifact MLflow téléchargé: %s", model_path)
                    break
            except Exception as e:
                last_error = e
                logger.warning(
                    "Échec du téléchargement de l'artifact %s: %s",
                    artifact_rel_path,
                    e,
                )

        if model_path is None:
            logger.error(
                "Impossible de télécharger un modèle depuis MLflow "
                "(essais: %s), dernière erreur: %s",
                artifact_candidates,
                last_error,
            )
            return False

        # 3) Instancier le prédicteur avec le fichier trouvé
        predictor = ProductPredictor(str(model_path))
        info = predictor.get_model_info()
        logger.info(
            "Modèle 'production' chargé: type=%s, classes=%s",
            info.get("model_type"),
            info.get("classes"),
        )
        return True

    except Exception as e:
        logger.error(
            "Erreur lors du chargement du modèle 'production' depuis MLflow: %s", e
        )
        return False


# =======================================================================
# Endpoints
# =======================================================================


@app.get("/", tags=["System"])
async def root():
    """Page d'accueil."""
    return {
        "message": "Bienvenue sur l'API Rakuten MLOps",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check."""
    db_status = "disconnected"
    try:
        check_database_connection()
        db_status = "connected"
    except Exception:
        pass

    model_status = predictor is not None
    overall_status = "healthy" if (db_status == "connected") else "degraded"

    return HealthResponse(
        status=overall_status,
        service="rakuten-mlops-api",
        version="1.0.0",
        timestamp=datetime.now(),
        database=db_status,
        model_loaded=model_status,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Infos sur le modèle chargé."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucun modèle chargé. Entraînez d'abord un modèle.",
        )

    info = predictor.get_model_info()

    return ModelInfoResponse(
        model_type=info["model_type"],
        model_path=info["model_path"],
        n_features=info.get("n_features"),
        n_classes=info.get("num_classes"),
        classes=info.get("classes"),
        has_label_encoder=info["has_label_encoder"],
    )


@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Endpoint de prédiction.

    La logique de feature engineering et de prédiction n'est pas
    implémentée dans cette version. Elle sera ajoutée dans une étape
    ultérieure par un autre membre de l'équipe.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Endpoint de prédiction non implémenté pour le moment. "
            "La logique de transformation des features et de prédiction "
            "sera ajoutée ultérieurement."
        ),
    )


@app.post("/training/", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """Lance l'entraînement complet."""
    start_time = time.time()

    try:
        logger.info(f"Démarrage de l'entraînement: {request.model_name}")

        check_database_connection()

        from src.utils.config import load_config
        from src.pipeline_steps.stage01_data_ingestion import (
            DataIngestionPipeline,
        )
        from src.pipeline_steps.stage02_data_validation import (
            DataValidationPipeline,
        )
        from src.pipeline_steps.stage03_data_transformation import (
            DataTransformationPipeline,
        )
        from src.pipeline_steps.stage04_model_training import (
            ModelTrainingPipeline,
        )
        from src.pipeline_steps.stage05_model_evaluation import (
            ModelEvaluationPipeline,
        )

        config = load_config()

        if request.model_name:
            config._config["model"]["name"] = request.model_name

        # Tracking URI : pris depuis la variable d'environnement MLFLOW_TRACKING_URI (docker-compose)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(request.experiment_name or "rakuten_classification")

        with mlflow.start_run(run_name=request.run_name) as run:
            # Tags pour suivi + promotion automatique
            if request.dag_run_id:
                mlflow.set_tag("dag_run_id", request.dag_run_id)
            mlflow.set_tag("api_triggered", "True")
            mlflow.set_tag("model_name", request.model_name)

            mlflow.log_param("model_name", request.model_name)
            mlflow.log_param("api_triggered", True)

            logger.info("Stage 1: Data Ingestion")
            stage1 = DataIngestionPipeline(config)
            X_train, y_train, X_test = stage1.run()

            logger.info("Stage 2: Data Validation")
            stage2 = DataValidationPipeline(config)
            validation_ok = stage2.run(X_train, y_train, X_test)

            if not validation_ok:
                raise ValueError("Validation échouée")

            logger.info("Stage 3: Data Transformation")
            stage3 = DataTransformationPipeline(config)
            (
                X_train_t,
                y_train_t,
                X_test_t,
                feature_pipeline,
                feature_mapping,
            ) = stage3.run(X_train, y_train, X_test)

            logger.info("Stage 4: Model Training")
            stage4 = ModelTrainingPipeline(config)
            model = stage4.run(X_train_t, y_train_t, feature_pipeline)

            # Construction du pipeline complet (features + modèle)
            logger.info("Construction du pipeline complet (features + modèle)")
            full_model = Pipeline(
                steps=[
                    ("features", feature_pipeline),
                    ("classifier", model),
                ]
            )

            full_model_path = "models/model_full_pipeline.joblib"
            to_save = {
                "model": full_model,
                "label_encoder": getattr(stage4.trainer, "label_encoder", None),
            }
            joblib.dump(to_save, full_model_path)
            logger.info(f"Pipeline complet sauvegardé dans {full_model_path}")

            # Logger les métriques CV dans MLflow
            if hasattr(stage4.trainer, "cv_scores_") and stage4.trainer.cv_scores_:
                cv_scores = stage4.trainer.cv_scores_
                if "f1_weighted" in cv_scores:
                    mlflow.log_metric(
                        "cv_f1_weighted_mean",
                        float(cv_scores["f1_weighted"]["mean"]),
                    )
                    mlflow.log_metric(
                        "cv_f1_weighted_std",
                        float(cv_scores["f1_weighted"]["std"]),
                    )
                    for idx, score in enumerate(
                        cv_scores["f1_weighted"]["scores"]
                    ):
                        mlflow.log_metric(
                            f"cv_f1_weighted_fold{idx + 1}", float(score)
                        )
                    logger.info(
                        "Métriques CV loggées: F1=%.4f",
                        cv_scores["f1_weighted"]["mean"],
                    )

            logger.info("Stage 5: Model Evaluation")

            # Note: y_test n'a pas de labels dans la BDD (données de compétition)
            logger.info("Pas de labels pour X_test (données de compétition)")

            stage5 = ModelEvaluationPipeline(config)
            metrics = stage5.run(
                model, X_test_t, None, stage4.trainer.label_encoder
            )

            model_name = config.model["name"]

            # Nom d'algo "propre" (tag informatif)
            raw_name = (model_name or "model").strip().lower()
            safe_name = re.sub(r"[^a-z0-9_]+", "_", raw_name)
            mlflow.set_tag("algo", safe_name)

            # Logger les métriques dans MLflow
            if metrics and isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                        logger.info("Métrique loggée: %s=%.4f", key, value)

            # Utiliser model.joblib qui est créé par stage04 pour le Registry
            model_path = "models/model.joblib"

            # Log du modèle comme artifact (modèle nu) - doit être fait pendant le run
            if Path(model_path).exists():
                saved_model = joblib.load(model_path)
                mlflow.sklearn.log_model(
                    saved_model,
                    artifact_path="model",
                )
                logger.info("Modèle loggé comme artifact dans MLflow")
            else:
                logger.warning(
                    "Fichier modèle introuvable pour le Registry: %s",
                    model_path,
                )

            # Capturer les variables nécessaires avant de sortir du contexte
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            # Logger aussi les fichiers .joblib comme artifacts simples
            try:
                if Path(model_path).exists():
                    mlflow.log_artifact(model_path)
                    logger.info("model.joblib loggé comme artifact")

                if Path(full_model_path).exists():
                    mlflow.log_artifact(full_model_path)
                    logger.info("model_full_pipeline.joblib loggé comme artifact")
            except Exception as e:
                logger.warning(f"Erreur lors du logging MLflow: {e}")

            # Mise à jour du predictor en mémoire avec le pipeline complet
            global predictor
            predictor = ProductPredictor(full_model_path)

            training_time = time.time() - start_time

            logger.info("Entraînement terminé en %.2fs", training_time)

            # Filtrer les métriques pour ne garder que les valeurs numériques
            filtered_metrics = None
            if metrics and isinstance(metrics, dict):
                filtered_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                    and k not in ["predictions", "dataset_name"]
                }

        # Enregistrer le modèle dans le MLflow Model Registry
        # IMPORTANT: Fait APRÈS la fin du run pour s'assurer que l'artifact est persisté
        if Path(model_path).exists():
            try:
                client = MlflowClient()

                # Créer le registered model s'il n'existe pas
                try:
                    client.create_registered_model(
                        REGISTERED_MODEL_NAME,
                        description="Modèle de classification Rakuten",
                    )
                    logger.info(
                        "Registered model '%s' créé", REGISTERED_MODEL_NAME
                    )
                except Exception as e:
                    if "ALREADY_EXISTS" in str(e):
                        logger.info(
                            "Registered model '%s' existe déjà",
                            REGISTERED_MODEL_NAME,
                        )
                    else:
                        raise

                # Créer une nouvelle version pour CE run
                # Le run est maintenant terminé, l'artifact devrait être persisté
                model_uri = f"runs:/{run_id}/model"
                mv = client.create_model_version(
                    name=REGISTERED_MODEL_NAME,
                    source=model_uri,
                    run_id=run_id,
                )
                logger.info(
                    "Modèle enregistré dans le Registry: %s version %s",
                    REGISTERED_MODEL_NAME,
                    mv.version,
                )

                # Taguer la version avec l'algo
                client.set_model_version_tag(
                    name=REGISTERED_MODEL_NAME,
                    version=mv.version,
                    key="algo",
                    value=safe_name,
                )
            except Exception as e:
                logger.error(
                    "Impossible d'enregistrer le modèle dans le Registry MLflow: %s",
                    e,
                )
                import traceback

                traceback.print_exc()
                # On continue quand même - le modèle est entraîné, juste pas enregistré
                # Mais on log en ERROR pour que ce soit visible

        return TrainingResponse(
            status="success",
            message=f"Modèle {model_name.upper()} entraîné avec succès",
            model_path=full_model_path,
            metrics=filtered_metrics,
            mlflow_run_id=run_id,
            mlflow_experiment_id=experiment_id,
            training_time=training_time,
        )

    except Exception as e:
        logger.error(f"Erreur dans l'entraînement: {e}")
        import traceback

        traceback.print_exc()

        return TrainingResponse(
            status="error",
            message=f"Erreur: {str(e)}",
            training_time=time.time() - start_time,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
