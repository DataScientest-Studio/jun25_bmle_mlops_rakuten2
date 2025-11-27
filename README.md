# Rakuten Multimodal MLOps Platform

> Plateforme MLOps complète pour la classification produits du challenge Rakuten, combinant pipelines texte/image, orchestration automatisée et serving via API et Streamlit.

## Présentation & TL;DR

- **Objectif** : entraîner, évaluer et servir un modèle multimodal (texte + image) pour classer les produits Rakuten (codes `prdtypecode`).
- **Dataset** : fichiers CSV officiels (`X_train_update.csv`, `Y_train_CVw08PX.csv`, `X_test_update.csv`) et images associées, versionnés via `snapshot.json`.
- **Technos clés** : PostgreSQL 16, pipelines scikit-learn/ResNet, PyTorch, XGBoost/LGBM, Streamlit, SHAP, Docker, MLflow, Airflow.

## Table des matières

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Pipeline ML](#pipeline-ml)
- [Configuration](#configuration)
- [Outils](#outils)
- [Ressources](#ressources)
- [Licence](#licence)

---

## Quick Start

### 1. Prérequis

- Docker & Docker Compose
- Python 3.10+
- (Optionnel) GPU pour accélérer les embeddings CNN

### 2. Installation

```bash
# Cloner le repo
git clone <repo-url>
cd jun25_bmle_mlops_rakuten2

# Option A : Partir des CSV bruts Rakuten
# 1. Placer les fichiers dans ./import/ pour qu'ils soient importés par ./postgres/09_copy_raw.sql pendant l'initialisation de la base de données
#    - X_train_update.csv
#    - Y_train_CVw08PX.csv  
#    - X_test_update.csv

# Option B : Télécharger le dump SQL pré-généré via DVC puis charger le dump
# dvc pull rakuten_dump.sql.dvc

# 2. Générer le hash et préparer les données
./init.sh

# Lancer les services
docker compose up -d
```

### 3. Accès aux services

| Service        | URL                                                    | Credentials       |
|----------------|--------------------------------------------------------|-------------------|
| PostgreSQL     | `postgres://mlops:mlops@localhost:5433/rakuten`        | mlops / mlops     |
| MLflow         | http://localhost:5000                                  | —                 |
| API FastAPI    | http://localhost:8000                                  | —                 |
| Airflow        | http://localhost:8081                                  | admin / admin123  |

---

## Architecture

```mermaid
flowchart LR
    subgraph Data
        CSV[Text CSV] -->|COPY| PG[(PostgreSQL)]
        Images[Images] --> DI
    end

    subgraph Pipeline
        DI(Ingestion) --> DV(Validation) --> DT(Transformation) --> MT(Training)
    end

    subgraph Services
        MLflow[MLflow:5000]
        API[API:8000]
        Airflow[Airflow:8081]
    end

    PG --> DI
    MT --> MLflow
    MT --> Artifacts[(models/)]
    Artifacts --> API
    Artifacts --> Streamlit
    Airflow --> API
    API --> Users((Users))
    Streamlit --> Users
```

#### Composants principaux

| Composant       | Description                                                                      |
|-----------------|----------------------------------------------------------------------------------|
| **PostgreSQL**  | Stockage des données produits (schéma `project.*`)                               |
| **MLflow**      | Tracking des expériences et Model Registry                                       |
| **FastAPI**     | Endpoint `/training/` pour lancer l'entraînement et `/predict/` pour l'inférence |
| **Airflow**     | Orchestration automatique (DAG `rakuten_training_pipeline`)                      |
| **Streamlit**   | Interface utilisateur interactive                                                |

---

#### Structure du projet

```
├── airflow/              # DAGs et config Airflow
├── api/                  # FastAPI (Dockerfile, main.py)
├── config/               # config.toml, labels_map.json
├── data/                 # Images (non versionné)
├── etl/                  # manifest_and_hash.py
├── import/               # CSV Rakuten
├── mlflow/               # Dockerfile MLflow
├── models/               # Modèles entraînés (.joblib)
├── postgres/             # Scripts SQL et init.sh
├── scripts/              # CLI (train_pipeline.py)
├── src/                  # Code source
│   ├── data/             # Chargement, sampling
│   ├── features/         # Text, Image, CNN
│   ├── models/           # ModelTrainer, predict
│   ├── pipeline_steps/   # Stages 01-05
│   ├── pipelines/        # text_pipeline, image_pipeline
│   └── utils/            # Config, logging, profiling
├── streamlit_app/        # App Streamlit
├── tools/                # Scripts utilitaires
├── docker-compose.yml
├── init.sh
└── snapshot.json         # Hash des données
```
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

| Endpoint       | Méthode | Description                          |
|----------------|---------|--------------------------------------|
| `/health`      | GET     | Health check                         |
| `/model/info`  | GET     | Infos sur le modèle chargé           |
| `/training/`   | POST    | Lance l'entraînement complet         |
| `/predict/`    | POST    | Prédiction *(non implémenté)*        |

Documentation Swagger : http://localhost:8000/docs

#### Airflow

DAG `rakuten_training_pipeline` :
1. Entraîne 3 modèles en parallèle (LR, XGBoost, LightGBM)
2. Promeut automatiquement le meilleur vers MLflow (alias `production`)

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

**Outputs** :
- `models/model.joblib` : modèle seul
- `models/model_full_pipeline.joblib` : pipeline complet (features + modèle)

---

### Évaluation & Explainability

- **Métriques** : Accuracy, F1 (macro/weighted), precision/recall
- **Confusion matrix** : valeurs, pourcentages, top erreurs
- **SHAP** : contributions par bloc (texte, CNN, stats)
- **Exports** : `results/metrics/` (JSON/CSV)



## Configuration

### `config/config.toml`

| Section | Description |
|---------|-------------|
| `[paths]` | Chemins CSV et images |
| `[features.text]` | Paramètres TF-IDF, nettoyage |
| `[features.image]` | CNN, stats, pixels |
| `[fusion.weights]` | Pondération des branches |
| `[model]` | Algorithme et hyperparamètres |
| `[cv]` | Validation croisée |
| `[sampling]` | Under/over-sampling |

### Variables d'environnement

- `MLFLOW_TRACKING_URI` : URI du serveur MLflow
- `DATABASE_HOST`, `DATABASE_PORT`, etc. : connexion PostgreSQL

---

## Outils

| Commande | Description |
|----------|-------------|
| `python tools/check_pipeline_branches.py` | Vérifie les branches actives |
| `python tools/test_pipeline_sample.py --sample-size 1500` | Run réduit avec profiling |
| `python tools/clear_cache.py --all` | Purge le cache |
| `python tools/shap_block_aggregation.py` | Agrège les valeurs SHAP |

---

## Ressources

- **Challenge Rakuten** : https://challengedata.ens.fr/participants/challenges/42/

---

## Licence

MIT License - voir [LICENSE](LICENSE)
