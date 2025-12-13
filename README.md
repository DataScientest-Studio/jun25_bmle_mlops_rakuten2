# Rakuten Multimodal MLOps Platform

## Présentation & TL;DR

- **Objectif** : entraîner, évaluer et servir un modèle multimodal (texte + image) pour classer les produits Rakuten (codes `prdtypecode`).
- **Dataset** : fichiers CSV officiels (`X_train_update.csv`, `Y_train_CVw08PX.csv`, `X_test_update.csv`) et images associées, versionnés via DVC.
- **Technos clés** : PostgreSQL 16, pipelines scikit-learn/ResNet, PyTorch, XGBoost/LGBM, Streamlit, SHAP, Docker, MLflow, Airflow, Prometheus, Grafana.

## Table des matières

- [Quick Start](#quick-start)
- [Pipeline ML](#pipeline-ml)
- [Architecture MLOps](#architecture-mlops)
- [Structure détaillée du projet](#structure-détaillée-du-projet)
- [Ressources](#ressources)
- [Licence](#licence)

---

## Quick Start

### 1. Prérequis

- Docker & Docker Compose
- (Optionnel) GPU pour accélérer les embeddings CNN

### 2. Installation

```bash
# Cloner le repo
git clone <repo-url>
cd jun25_bmle_mlops_rakuten2

# Créer un environnement uv et l'activer
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

# Initialiser les variables d'environnement à partir du template et les modifier selon besoin
cp .env.example .env

# Récupération des images
dvc pull ./data/images.dvc

# Récupération des textes via DVC dans ./data/raw/
dvc pull ./data/raw/X_train_update.csv.dvc
dvc pull ./data/raw/Y_train_CVw08PX.csv.dvc
dvc pull ./data/raw/X_test_update.csv.dvc

# Pour accélérer le process local, récupérer le dump SQL pré-généré via DVC puis charger le dump
# dvc pull ./postgres/dump/rakuten_dump.sql.dvc
# Configurer la valeur LOAD_DB_DUMP=1 dans .env

# Lancer les services
docker compose up -d
```

### 3. Accès aux services

| Service        | URL                                                    | Credentials (default) |
|----------------|--------------------------------------------------------|-----------------------|
| PostgreSQL     | `postgres://mlops:mlops@localhost:5433/rakuten`        | mlops / mlops         |
| MLflow         | http://localhost:5000                                  | —                     |
| API FastAPI    | http://localhost:8000                                  | —                     |
| Airflow        | http://localhost:8080                                  | admin / admin123      |
| Streamlit      | http://localhost:8501                                  | —                     |
| Prometheus     | http://localhost:9090                                  | —                     |
| Grafana        | http://localhost:3000                                  | admin / admin         |
| Node Exporter  | http://localhost:9100                                  | —                     |

---

## Pipeline ML

Exécution via `python scripts/train_pipeline.py` ou via l'API/Airflow.

| Étape | Module | Description |
|-------|--------|-------------|
| 1     | `stage01_data_ingestion.py` | Chargement depuis PostgreSQL/CSV |
| 2     | `stage02_data_validation.py` | Validation schéma et qualité |
| 3     | `stage03_data_transformation.py` | Feature engineering multimodal |
| 4     | `stage04_model_training.py` | Entraînement + CV |
| 5     | `stage05_model_evaluation.py` | Métriques, confusion matrix, SHAP |

![Pipeline ML](streamlit_app/assets/modelisation.drawio.svg)

**Options CLI** :
```bash
python scripts/train_pipeline.py --skip-validation --cv --evaluate-on-train
```

---

### Feature Engineering

#### Texte

- **Nettoyage** : normalisation Unicode, traduction, stemming
- **TF-IDF** : n-grams 1-2, `max_features` configurable
- **Stats** : longueur titre, indicateur description, détection langue

#### Images

- **Pixels** : aplatissement + PCA/SVD optionnel
- **Stats** : occupancy, entropy, gradients, colorimétrie
- **CNN** : embeddings ResNet18/50/101 ou ViT

#### Fusion

Combinaison via `FeatureUnion` avec pondérations configurables dans `config/config.toml` section `[fusion.weights]`.

---

### Modélisation

**Algorithmes supportés** : Logistic Regression, Linear SVC, XGBoost, LightGBM

**Configuration** : `config/config.toml` section `[model]`

**Outputs** (versionnés avec timestamp `YYYYMMDD_HHMMSS`) :
- `models/{model_name}_final_{timestamp}.joblib` : modèle seul
- `models/{model_name}_full_pipeline_{timestamp}.joblib` : pipeline complet (features + modèle)
- `models/{model_name}_feature_pipeline_{timestamp}.joblib` : pipeline de features seul

---

### Évaluation & Explainability

- **Métriques** : Accuracy, F1 (macro/weighted), precision/recall
- **Confusion matrix** : valeurs, pourcentages, top erreurs
- **SHAP** : contributions par bloc (texte, CNN, stats)
- **Exports** : `results/metrics/` (JSON/CSV)

---

### Configuration

Dans `config/config.toml` :

| Section | Description |
|---------|-------------|
| `[paths]` | Chemins CSV et images |
| `[features.text]` | Paramètres TF-IDF, nettoyage |
| `[features.image]` | CNN, stats, pixels |
| `[fusion.weights]` | Pondération des branches |
| `[model]` | Algorithme et hyperparamètres |
| `[cv]` | Validation croisée |
| `[sampling]` | Under/over-sampling |

---

### Outils

| Commande | Description |
|----------|-------------|
| `python tools/check_pipeline_branches.py` | Vérifie les branches actives |
| `python tools/test_pipeline_sample.py --sample-size 1500` | Run réduit avec profiling |
| `python tools/clear_cache.py --all` | Purge le cache |
| `python tools/shap_block_aggregation.py` | Agrège les valeurs SHAP |

---

## Architecture MLOps

![Architecture globale](global-architecture.drawio.svg)

---

#### Composants principaux

| Composant        | Description                                                                      |
|------------------|----------------------------------------------------------------------------------|
| **PostgreSQL**   | Stockage des données produits (schéma `project.*`)                               |
| **MLflow**       | Tracking des expériences et Model Registry                                       |
| **FastAPI**      | Endpoint `/training/` pour lancer l'entraînement et `/predict/` pour l'inférence |
| **Airflow**      | Orchestration automatique (DAG `rakuten_training_pipeline`)                      |
| **Streamlit**    | Interface utilisateur interactive pour visualisation et démonstration            |
| **Prometheus**   | Collecte des métriques système et applicatives                                   |
| **Grafana**      | Visualisation des métriques et dashboards de monitoring                          |
| **Node Exporter**| Export des métriques système (CPU, mémoire, disque)                              |

---

### Micro services

#### PostgreSQL

Base de données avec schéma `project` contenant les produits Rakuten.

```sql
-- Vérification
SELECT COUNT(*) FROM project.items;
SELECT prdtypecode, COUNT(*) FROM project.items GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```

#### MLflow

- **Tracking** : métriques, paramètres, artifacts
- **Registry** : versioning des modèles avec alias `production`
- **Backend** : PostgreSQL (`mlflow` database)

#### API FastAPI

| Endpoint         | Méthode | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `/`              | GET     | Page d'accueil avec liens                                |
| `/health`        | GET     | Health check (DB + modèle)                               |
| `/model/info`    | GET     | Infos sur le modèle chargé                               |
| `/training/`     | POST    | Lance l'entraînement complet                             |
| `/predict/`      | POST    | Prédiction via `multipart/form-data` (image + texte)     |
| `/reload-model/` | POST    | Recharge le modèle `production` depuis MLflow            |

**Exemple d'appel `/predict/`** :
```bash
# Le contenu de l'image est envoyé au format multipart/form-data
curl -X POST "http://localhost:8000/predict/" \
     -F "designation=iPhone 13" \
     -F "description=Smartphone Apple" \
     -F "image=@./mon_image.jpg"
```

Documentation Swagger : http://localhost:8000/docs

#### Airflow

DAG `rakuten_training_pipeline` :
1. Entraîne 3 modèles en parallèle (LR, XGBoost, LightGBM)
2. Promeut automatiquement le meilleur vers MLflow (alias `production`)
3. Recharge le modèle dans l'API via `/reload-model/`
4. Log de succès

**Configuration** :
- Schedule : une fois par jour (`0 0 * * *`)
- Métrique de sélection : `cv_f1_weighted_mean`
- Les modèles sont entraînés via l'API FastAPI (`/training/`)

#### Streamlit

Application web interactive pour présenter le projet et fournir une interface d'utilisation du modèle pour inférence.

**Accès** : http://localhost:8501

#### Monitoring (Prometheus + Grafana)

Stack de monitoring pour surveiller les performances et la santé du système :

- **Node Exporter** : collecte les métriques système (CPU, mémoire, disque, réseau)
- **Prometheus** : agrège et stocke les métriques avec un système de requêtes PromQL
- **Grafana** : visualisation via dashboards préconfigurés

**Configuration** :
- Dashboards Grafana : `grafana/dashboards/`
- Configuration Prometheus : `prometheus/prometheus.yml`

---

## Structure détaillée du projet

```
├── airflow/              # DAGs et config Airflow
│   └── dags/             # Définition du DAG rakuten_training_pipeline
├── api/                  # FastAPI
│   ├── src/              # Code source (main.py, schemas.py)
│   └── Dockerfile        # Image Docker pour l'API
├── config/               # Configuration
│   ├── config.toml       # Configuration principale du pipeline
│   ├── labels_map.json   # Mapping prdtypecode → labels lisibles
│   ├── theme_map.json    # Mapping thématique des catégories
│   └── translate_map_starter_from_cleaned.json  # Dictionnaire de traduction
├── data/                 # Images et textes RAW (non versionné, via DVC)
│   ├── raw/              # CSV sources
│   └── images/           # Images train/test
├── grafana/              # Configuration Grafana
│   ├── dashboards/       # Dashboards JSON préconfigurés
│   └── provisioning/     # Auto-provisioning des datasources
├── mlflow/               # Dockerfile MLflow
├── models/               # Modèles entraînés (.joblib versionnés)
├── postgres/             # Base de données
│   ├── sql/              # Scripts SQL d'initialisation
│   ├── dump/             # Dumps SQL (optionnel, via DVC)
│   └── init.sh           # Script d'init du container
├── prometheus/           # Configuration Prometheus
│   └── prometheus.yml    # Targets de scraping
├── references/           # Documentation et références
├── reports/              # Rapports générés
├── scripts/              # CLI et scripts d'exécution
│   ├── train_pipeline.py # Point d'entrée principal
│   ├── predict.py        # Script de prédiction
│   └── test_api.py       # Tests de l'API
├── src/                  # Code source
│   ├── data/             # Chargement, sampling
│   ├── features/         # Text, Image, CNN
│   ├── models/           # ModelTrainer, predict
│   ├── pipeline_steps/   # Stages 01-05
│   ├── pipelines/        # text_pipeline, image_pipeline
│   ├── utils/            # Config, logging, profiling
│   └── visualization/    # Data visualization
├── streamlit_app/        # App Streamlit
│   ├── app.py            # Application principale
│   ├── config.py         # Configuration (titre, équipe)
│   ├── tabs/             # Onglets (intro, architecture, modelisation, demonstration, conclusion)
│   ├── assets/           # Images, SVG et ressources statiques
│   └── Dockerfile        # Image Docker pour Streamlit
├── tools/                # Scripts utilitaires
├── .env.example          # Template des variables d'environnement
├── docker-compose.yml    # Infrastructure as code : configuration des micro-services
├── LICENSE               # License du projet
└── requirements.txt      # Dépendances Python du projet
```

---

## Ressources

- **Challenge Rakuten** : https://challengedata.ens.fr/participants/challenges/42/
- **8 MLOps Best Practices for Scalable, Production-Ready ML Systems** : https://www.azilen.com/blog/mlops-best-practices/

---

## Licence

MIT License - voir [LICENSE](LICENSE)
