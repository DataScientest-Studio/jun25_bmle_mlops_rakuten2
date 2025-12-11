"""
DAG Airflow - Pipeline d'entraînement Rakuten
=============================================

Ce DAG orchestre l'entraînement des modèles de classification Rakuten
et promeut automatiquement le meilleur modèle dans MLflow.
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator
from mlflow.tracking import MlflowClient

# Configuration du logger
logger = logging.getLogger(__name__)

# Constantes MLflow
MLFLOW_EXPERIMENT_NAME = "rakuten_classification"
REGISTERED_MODEL_NAME = "model"          # même nom que dans le Model Registry
BEST_METRIC = "cv_f1_weighted_mean"      # métrique de sélection

# Arguments par défaut du DAG
default_args = {
    "owner": "datascientest",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 14),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(days=1),
}

# Définition du DAG
dag = DAG(
    dag_id="rakuten_training_pipeline",
    default_args=default_args,
    description="Pipeline complet d'entraînement des modèles Rakuten",
    schedule_interval="*/15 * * * *",  # exécution toutes les 15 minutes
    catchup=False,
    max_active_runs=1,                  # éviter plusieurs runs en parallèle
    tags=["rakuten", "ml", "training"],
)

# ============================================
# TÂCHE 1 : Entraîner Logistic Regression
# ============================================
train_lr = SimpleHttpOperator(
    task_id="train_logistic_regression",
    http_conn_id="rakuten_api",
    endpoint="/training/",
    method="POST",
    # on ajoute le dag_run_id dans le payload (template Jinja)
    data=(
        '{"model_name": "lr", '
        '"experiment_name": "rakuten_classification", '
        '"run_name": "airflow_lr", '
        '"dag_run_id": "{{ dag_run.run_id }}"}'
    ),
    headers={"Content-Type": "application/json"},
    response_check=lambda response: "success" in response.text.lower(),
    log_response=True,
    # Timeout de 8 heures pour permettre l'entraînement complet (peut prendre jusqu'à 6h)
    extra_options={"timeout": 28800},  # 8 heures en secondes
    execution_timeout=timedelta(hours=8, minutes=30),  # Légèrement supérieur au timeout HTTP
    dag=dag,
)

# ============================================
# TÂCHE 2 : Entraîner XGBoost
# ============================================
train_xgb = SimpleHttpOperator(
    task_id="train_xgboost",
    http_conn_id="rakuten_api",
    endpoint="/training/",
    method="POST",
    data=(
        '{"model_name": "xgb", '
        '"experiment_name": "rakuten_classification", '
        '"run_name": "airflow_xgb", '
        '"dag_run_id": "{{ dag_run.run_id }}"}'
    ),
    headers={"Content-Type": "application/json"},
    response_check=lambda response: "success" in response.text.lower(),
    log_response=True,
    # Timeout de 8 heures pour permettre l'entraînement complet (peut prendre jusqu'à 6h)
    extra_options={"timeout": 28800},  # 8 heures en secondes
    execution_timeout=timedelta(hours=8, minutes=30),  # Légèrement supérieur au timeout HTTP
    dag=dag,
)

# ============================================
# TÂCHE 3 : Entraîner LightGBM
# ============================================
train_lgbm = SimpleHttpOperator(
    task_id="train_lightgbm",
    http_conn_id="rakuten_api",
    endpoint="/training/",
    method="POST",
    data=(
        '{"model_name": "lgbm", '
        '"experiment_name": "rakuten_classification", '
        '"run_name": "airflow_lgbm", '
        '"dag_run_id": "{{ dag_run.run_id }}"}'
    ),
    headers={"Content-Type": "application/json"},
    response_check=lambda response: "success" in response.text.lower(),
    log_response=True,
    # Timeout de 8 heures pour permettre l'entraînement complet (peut prendre jusqu'à 6h)
    extra_options={"timeout": 28800},  # 8 heures en secondes
    execution_timeout=timedelta(hours=8, minutes=30),  # Légèrement supérieur au timeout HTTP
    dag=dag,
)

# ============================================
# TÂCHE 4 : Promotion auto du meilleur modèle
# ============================================
def promote_best_model(**context):
    """
    Sélectionne le meilleur run de ce dag_run (sur BEST_METRIC)
    et assigne l'alias 'production' à la version correspondante
    dans le Model Registry MLflow.
    """
    client = MlflowClient()

    # 1. Récupérer l'expérience
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment {MLFLOW_EXPERIMENT_NAME} not found")

    # 2. Filtrer les runs du même dag_run_id
    dag_run_id = context["dag_run"].run_id
    filter_string = f"tags.dag_run_id = '{dag_run_id}'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=[f"metrics.{BEST_METRIC} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No runs found for experiment {MLFLOW_EXPERIMENT_NAME} "
            f"and filter {filter_string}"
        )

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_score = best_run.data.metrics.get(BEST_METRIC)

    logger.info(
        "Best run for DAG run %s: %s with %s=%s",
        dag_run_id,
        best_run_id,
        BEST_METRIC,
        best_score,
    )

    # 3. Retrouver la version de modèle enregistrée associée à ce run
    model_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    best_version = None
    for mv in model_versions:
        if mv.run_id == best_run_id:
            best_version = mv
            break

    if best_version is None:
        raise ValueError(
            f"No registered model version found for run_id={best_run_id} "
            f"and model '{REGISTERED_MODEL_NAME}'"
        )

    logger.info(
        "Setting alias 'production' on model '%s' version %s",
        REGISTERED_MODEL_NAME,
        best_version.version,
    )

    # 4. Assigner l'alias 'production' à la meilleure version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="production",
        version=best_version.version,
    )

    # 5. Tag pour savoir d'où vient la sélection
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=best_version.version,
        key="selected_by",
        value="airflow_auto_promotion",
    )


promote_model = PythonOperator(
    task_id="promote_best_model",
    python_callable=promote_best_model,
    provide_context=True,
    dag=dag,
)

# ============================================
# TÂCHE 5 : Log de succès
# ============================================
def log_pipeline_success(**context):
    """Log le succès du pipeline complet."""
    logger.info("=" * 60)
    logger.info("PIPELINE RAKUTEN TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)
    logger.info("Logistic Regression entraînée")
    logger.info("XGBoost entraînée")
    logger.info("LightGBM entraînée")
    logger.info(
        "Meilleur modèle promu dans le Model Registry MLflow (alias 'production')"
    )
    logger.info("Consultez MLflow pour les résultats : http://localhost:5000")
    logger.info("=" * 60)


log_success = PythonOperator(
    task_id="log_pipeline_success",
    python_callable=log_pipeline_success,
    provide_context=True,
    dag=dag,
)

# ============================================
# DÉFINITION DU FLUX D'EXÉCUTION
# ============================================
# Les 3 modèles s'entraînent en parallèle -> promotion -> log
[train_lr, train_xgb, train_lgbm] >> promote_model >> log_success

