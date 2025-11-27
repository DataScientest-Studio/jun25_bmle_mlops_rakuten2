# Rakuten Multimodal MLOps Platform

## Présentation & TL;DR

- **Objectif** : entraîner, évaluer et servir un modèle multimodal (texte + image) pour classer les produits Rakuten (codes `prdtypecode`).
- **Dataset** : fichiers CSV officiels (`X_train_update.csv`, `Y_train_CVw08PX.csv`, `X_test_update.csv`) et images associées, versionnés via `snapshot.json`.
- **Technos clés** : PostgreSQL 16, pipelines scikit-learn/ResNet, PyTorch, XGBoost/LGBM, Streamlit, SHAP, Docker, MLflow, Airflow.

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

# Option A : Partir des CSV bruts Rakuten
# A1. Placer les fichiers dans ./data/raw/ pour qu'ils soient importés par ./postgres/09_copy_raw.sql pendant l'initialisation de la base de données
#    - X_train_update.csv
#    - Y_train_CVw08PX.csv  
#    - X_test_update.csv
# A2. Configurer la valeur LOAD_DB_DUMP=0 dans .env
# A3. Générer le hash et préparer les données
# ./init.sh

# Option B : Télécharger le dump SQL pré-généré via DVC puis charger le dump
# B1. dvc pull ./postgres/dump/rakuten_dump.sql.dvc
# B2. Configurer la valeur LOAD_DB_DUMP=1 dans .env

# Lancer les services
docker compose up -d
```

### 3. Accès aux services

| Service        | URL                                                    | Credentials (default) |
|----------------|--------------------------------------------------------|-----------------------|
| PostgreSQL     | `postgres://mlops:mlops@localhost:5433/rakuten`        | mlops / mlops         |
| MLflow         | http://localhost:5000                                  | —                     |
| API FastAPI    | http://localhost:8000                                  | —                     |
| Airflow        | http://localhost:8081                                  | admin / admin123      |

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

```mermaid
flowchart LR
    subgraph Data Pipeline
        CSV[Text CSV files] --> DI
        DI(Ingestion) --> PG[(PostgreSQL)]
        Images[Images] --> DI
    end

    subgraph ML Pipeline
        PG --> DV(Validation) --> DT(Transformation) --> MT(Training)
    end

    subgraph Services
        MLflow[MLflow:5000]
        API[API:8000]
        Airflow[Airflow:8081]
    end

    MT --> MLflow
    MT --> Artifacts[(models/)]
    Artifacts --> API
    Artifacts --> Streamlit
    API --> Airflow
    Airflow --> MT
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

## Structure détaillée du projet

```
├── airflow/              # DAGs et config Airflow
├── api/                  # FastAPI (Dockerfile, main.py)
├── config/               # config.toml, labels_map.json
├── data/                 # Images et textes RAW (non versionné)
├── etl/                  # manifest_and_hash.py
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
│   ├── utils/            # Config, logging, profiling
│   └── visualization/    # Data visualization
├── streamlit_app/        # App Streamlit
├── tools/                # Scripts utilitaires
├── .env.example          # Template des variables d'environnement
├── docker-compose.yml    # Infrastructure as code : configuration des micro-services
├── init.sh               # Script d'initialisation du hash des données
├── LICENSE               # License du projet
├── requirements.txt      # Dépendences Python du projet
└── snapshot.json         # Hash des données
```

---

## Ressources

- **Challenge Rakuten** : https://challengedata.ens.fr/participants/challenges/42/

---

## Licence

MIT License - voir [LICENSE](LICENSE)
