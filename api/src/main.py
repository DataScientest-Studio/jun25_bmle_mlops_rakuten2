"""
API FastAPI pour le projet Rakuten MLOps.
=========================================

Cette API expose deux endpoints principaux:
- POST /training/ : Lance l'entraînement d'un modèle
- POST /predict/ : Fait des prédictions sur de nouvelles données (multipart/form-data)

Démarrage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd


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
from typing import Optional
import re  # pour nettoyer le nom de modèle

import psycopg2
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
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
    * **POST /predict/** - Prédit la catégorie d'un produit (multipart/form-data)
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

    # Look for timestamped full pipeline files (format: {model_name}_full_pipeline_{timestamp}.joblib)
    model_files = list(models_dir.glob("*_full_pipeline_*.joblib"))

    if not model_files:
        logger.warning("Aucun modèle versionné trouvé (format: *_full_pipeline_*.joblib)")
        return

    # Sort by filename (which includes timestamp) to get the most recent
    latest_model = max(model_files, key=lambda p: p.name)

    logger.info(f"Chargement du modèle local (le plus récent): {latest_model}")

    predictor = ProductPredictor(str(latest_model))

    logger.info(f"Modèle chargé: {predictor.get_model_info()['model_type']}")


def load_production_model_from_mlflow() -> bool:
    """
    Charge le modèle avec l'alias 'production' depuis le Model Registry MLflow.
    Les fichiers de modèle sont versionnés avec des timestamps (format: YYYYMMDD_HHMMSS).
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

        # Get model type from version tags
        model_type = None
        try:
            tags = client.get_model_version(mv.name, mv.version).tags
            model_type = tags.get("algo", "").replace("_", "")
        except Exception as e:
            logger.warning(f"Impossible de récupérer les tags du modèle: {e}")

        if not model_type:
            logger.error("Type de modèle non trouvé dans les tags MLflow")
            return False

        # Try to download the timestamped full_pipeline file from artifacts
        # The file is logged as an artifact at the root level (line 704 in training code)
        try:
            # First try to download from root artifacts (where timestamped files are logged)
            artifacts_dir = client.download_artifacts(
                run_id=mv.run_id,
                path=".",  # Root of artifacts
                dst_path=str(local_models_dir),
            )
            
            # Look for the timestamped full_pipeline file
            artifacts_path = Path(artifacts_dir)
            pattern = f"{model_type}_full_pipeline_*.joblib"
            matching_files = list(artifacts_path.rglob(pattern))  # rglob for recursive search
            
            if matching_files:
                model_path = max(matching_files, key=lambda p: p.name)
                logger.info("Modèle timestampé trouvé dans artifacts: %s", model_path)
            else:
                # Fallback: Try downloading from models directory
                logger.info("Fichier non trouvé à la racine, tentative dans models/...")
                artifacts_dir = client.download_artifacts(
                    run_id=mv.run_id,
                    path="models",
                    dst_path=str(local_models_dir),
                )
                artifacts_path = Path(artifacts_dir)
                if artifacts_path.is_dir():
                    matching_files = list(artifacts_path.glob(pattern))
                    if matching_files:
                        model_path = max(matching_files, key=lambda p: p.name)
                        logger.info("Modèle timestampé trouvé dans models/: %s", model_path)
                    else:
                        raise FileNotFoundError(
                            f"Aucun fichier modèle trouvé correspondant au pattern: {pattern}. "
                            f"Vérifiez que le modèle a été entraîné avec le format timestampé."
                        )
                else:
                    model_path = artifacts_path
                    logger.info("Artifact MLflow téléchargé: %s", model_path)

        except Exception as e:
            logger.error(
                "Erreur lors du téléchargement des artifacts MLflow: %s", e
            )
            import traceback
            logger.error(traceback.format_exc())
            return False

        if not model_path or not Path(model_path).exists():
            logger.error("Fichier modèle introuvable: %s", model_path)
            return False

        # 3) Instancier le prédicteur avec le fichier trouvé
        predictor = ProductPredictor(str(model_path))
        info = predictor.get_model_info()
        logger.info(
            "Modèle 'production' chargé: type=%s, classes=%s, fichier=%s",
            info.get("model_type"),
            info.get("classes"),
            model_path,
        )
        return True

    except Exception as e:
        logger.error(
            "Erreur lors du chargement du modèle 'production' depuis MLflow: %s", e
        )
        import traceback
        logger.error(traceback.format_exc())
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


@app.post("/reload-model/", tags=["Admin"])
async def reload_model():
    """
    Recharge le modèle 'production' depuis MLflow.
    
    Utilisé par Airflow après la promotion d'un nouveau modèle
    pour que l'API utilise immédiatement la nouvelle version.
    """
    global predictor
    
    logger.info("Rechargement du modèle de production demandé...")
    
    loaded = load_production_model_from_mlflow()
    
    if loaded:
        model_info = predictor.get_model_info() if predictor else {}
        logger.info(
            "Modèle de production rechargé avec succès: %s",
            model_info.get("model_type", "unknown")
        )
        return {
            "status": "success",
            "message": "Modèle de production rechargé avec succès",
            "model_type": model_info.get("model_type"),
            "model_path": model_info.get("model_path"),
        }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Aucun modèle avec l'alias 'production' trouvé dans MLflow",
    )


@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    designation: str = Form(..., description="Désignation du produit"),
    description: Optional[str] = Form(None, description="Description du produit"),
    image: UploadFile = File(..., description="Image du produit (JPEG, PNG)")
):
    """
    Endpoint de prédiction avec upload de fichier (multipart/form-data).
    
    Utilise multipart/form-data pour une transmission efficace des images.

    Exemple d'utilisation:
        curl -X POST "http://localhost:8000/predict/" \\
             -F "designation=iPhone 13" \\
             -F "description=Smartphone Apple" \\
             -F "image=@/path/to/image.jpg"
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucun modèle chargé. Entraînez d'abord un modèle ou attendez le chargement du modèle de production.",
        )
    
    # Vérifier le type de fichier
    if image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Type de fichier non supporté: {image.content_type}. Utilisez JPEG ou PNG.",
        )
    
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la lecture de l'image: {str(e)}",
        )
    
    # Créer un DataFrame avec les colonnes attendues par le pipeline
    df = pd.DataFrame({
        "designation": [designation],
        "description": [description] if description else [None],
        "image_binary": [image_bytes],  # Bytes directs, pas base64
    })
    
    # Utiliser predict_with_confidence pour obtenir prédiction, confiance et top classes
    results = predictor.predict_with_confidence(df)
    
    # Extraire le premier résultat (on ne prédit qu'un seul produit)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aucun résultat de prédiction retourné.",
        )
    
    result = results[0]
    
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result.get("confidence"),
        top_classes=result.get("top_classes"),
    )

@app.post("/training/", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """Lance l'entraînement complet."""
    start_time = time.time()

    try:
        logger.info(f"Démarrage de l'entraînement: {request.model_name}")

        # Generate timestamp once for consistent versioning across all files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        # Store timestamp in config for use in stage04
        config._config["training_timestamp"] = timestamp

        # Tracking URI : pris depuis la variable d'environnement MLFLOW_TRACKING_URI (docker-compose)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(request.experiment_name or "rakuten_classification")

        # Enable MLflow autolog for sklearn to automatically log parameters and metrics
        # log_models=False because we manually log the full pipeline (features + classifier)
        # This avoids warnings about transformers missing predict() method
        mlflow.sklearn.autolog(
            log_input_examples=False,  # We log samples manually
            log_model_signatures=False,  # Signature is set manually on full pipeline
            log_models=False,  # Disabled: we manually log the full pipeline below
            registered_model_name=None,  # We register manually after training
        )

        with mlflow.start_run(run_name=request.run_name) as run:
            # Tags pour suivi + promotion automatique
            if request.dag_run_id:
                mlflow.set_tag("dag_run_id", request.dag_run_id)
            mlflow.set_tag("api_triggered", "True")
            mlflow.set_tag("model_name", request.model_name)

            mlflow.log_param("model_name", request.model_name)
            mlflow.log_param("api_triggered", True)

            logger.info("Stage 1: Data Ingestion")
            stage1 = DataIngestionPipeline(config, source="postgres", db_config=DATABASE_CONFIG)
            X_train, y_train, X_test = stage1.run()

            # Log input data information
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("train_features", X_train.shape[1] if hasattr(X_train, 'shape') else 'unknown')
            mlflow.log_param("num_classes", len(y_train.unique()) if hasattr(y_train, 'unique') else 'unknown')

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
            
            # Get raw inputs from stage3 (after image loading: designation, description, image_binary)
            # This matches the API inputs
            X_train_raw = getattr(stage3, 'X_train_raw', None)
            if X_train_raw is None:
                logger.warning("X_train_raw not available from stage3, using X_train")
                X_train_raw = X_train.copy()

            # Log transformed data information
            mlflow.log_param("transformed_train_features", X_train_t.shape[1] if hasattr(X_train_t, 'shape') else 'unknown')
            mlflow.log_param("transformed_test_features", X_test_t.shape[1] if hasattr(X_test_t, 'shape') else 'unknown')
            
            # Log sample input data as artifact for reference
            try:
                import numpy as np
                import pandas as pd
                sample_input = X_train_t[:5] if len(X_train_t) >= 5 else X_train_t
                # Save as CSV for easy inspection
                if isinstance(sample_input, np.ndarray):
                    sample_df = pd.DataFrame(sample_input)
                else:
                    sample_df = pd.DataFrame(sample_input) if hasattr(sample_input, 'to_frame') else sample_input
                artifacts_dir = Path("artifacts")
                artifacts_dir.mkdir(exist_ok=True)
                sample_path = artifacts_dir / "train_input_sample.csv"
                sample_df.to_csv(sample_path, index=False)
                mlflow.log_artifact(str(sample_path), artifact_path="data_samples")
                logger.info("Échantillon de données d'entraînement loggé")
            except Exception as e:
                logger.warning(f"Erreur lors du logging de l'échantillon d'entrée: {e}")

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

            # Use the timestamp generated at the start of training
            model_name = config.model["name"]
            
            # Log the full pipeline to MLflow (this is the production-ready model)
            logger.info("Enregistrement du pipeline complet dans MLflow...")
            try:
                # Create sample input for signature matching API inputs:
                # designation (str), description (str), image_binary (bytes)
                api_columns = ['designation', 'description', 'image_binary']
                available_cols = [c for c in api_columns if c in X_train_raw.columns]
                sample_input = X_train_raw[available_cols].iloc[:1].copy()
                logger.info(f"Sample input columns for MLflow signature: {list(sample_input.columns)}")
                
                # Log the full pipeline with signature
                mlflow.sklearn.log_model(
                    full_model,
                    name="full_pipeline",
                    input_example=sample_input,
                    registered_model_name=None,  # We'll register manually
                )
                logger.info("Pipeline complet loggé dans MLflow avec signature (inputs bruts: description, designation, image_binary)")
            except Exception as e:
                logger.warning(f"Erreur lors du logging du pipeline complet: {e}")
            
            # Save full pipeline with timestamp versioning
            full_model_path = f"models/{model_name}_full_pipeline_{timestamp}.joblib"
            to_save = {
                "model": full_model,
                "label_encoder": getattr(stage4.trainer, "label_encoder", None),
                "timestamp": timestamp,
                "model_name": model_name,
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

            # Log sample predictions/outputs for reference
            try:
                import numpy as np
                import pandas as pd
                sample_predictions = model.predict(X_test_t[:10]) if len(X_test_t) >= 10 else model.predict(X_test_t)
                # Save predictions as CSV
                predictions_df = pd.DataFrame({
                    "predictions": sample_predictions,
                    "sample_indices": range(len(sample_predictions))
                })
                artifacts_dir = Path("artifacts")
                artifacts_dir.mkdir(exist_ok=True)
                predictions_path = artifacts_dir / "test_predictions_sample.csv"
                predictions_df.to_csv(predictions_path, index=False)
                mlflow.log_artifact(str(predictions_path), artifact_path="data_samples")
                logger.info("Échantillon de prédictions loggé dans MLflow")
            except Exception as e:
                logger.warning(f"Erreur lors du logging des prédictions: {e}")

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

            # Construct model path with timestamp versioning
            model_path_template = config.paths.get(
                "model_out",
                "models/{kind}_{phase}_{timestamp}.joblib"
            )
            model_path = model_path_template.replace("{kind}", model_name)
            model_path = model_path.replace("{phase}", "final")
            model_path = model_path.replace("{timestamp}", timestamp)

            # Save feature pipeline as artifact (not as model - it's a transformer without predict)
            logger.info("Sauvegarde du pipeline de features comme artifact...")
            try:
                feature_pipeline_path = f"models/{model_name}_feature_pipeline_{timestamp}.joblib"
                joblib.dump(feature_pipeline, feature_pipeline_path)
                mlflow.log_artifact(feature_pipeline_path, artifact_path="feature_pipeline")
                logger.info(f"Pipeline de features sauvegardé: {feature_pipeline_path}")
            except Exception as e:
                logger.warning(f"Erreur lors de la sauvegarde du pipeline de features: {e}")

            # Log du modèle comme artifact (modèle nu) - doit être fait pendant le run
            if Path(model_path).exists():
                saved_data = joblib.load(model_path)
                # Extract model from dict if saved as dict structure
                if isinstance(saved_data, dict) and 'model' in saved_data:
                    classifier_model = saved_data['model']
                else:
                    classifier_model = saved_data
                
                try:
                    # NOTE: The standalone model (classifier only) expects TRANSFORMED features (X_train_t)
                    # This is different from full_pipeline which expects raw inputs
                    sample_input = X_train_t[:1] if len(X_train_t) >= 1 else X_train_t
                    # Convert to float64 to avoid MLflow schema warning about integer columns
                    if hasattr(sample_input, 'astype'):
                        sample_input = sample_input.astype('float64')
                    mlflow.sklearn.log_model(
                        classifier_model,
                        name="model",
                        input_example=sample_input,
                    )
                    logger.info("Modèle loggé comme artifact dans MLflow (inputs: features transformées)")
                except Exception as e:
                    logger.warning(f"Erreur lors du logging du modèle: {e}")
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
                    logger.info("Modèle versionné loggé comme artifact: %s", model_path)

                if Path(full_model_path).exists():
                    mlflow.log_artifact(full_model_path)
                    logger.info("Pipeline complet versionné loggé comme artifact: %s", full_model_path)
                
                # Log timestamp as parameter for easy filtering
                mlflow.log_param("model_timestamp", timestamp)
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
