FROM python:3.11-slim

WORKDIR /mlflow

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Installation de MLflow et du driver PostgreSQL
RUN pip install \
    mlflow==3.6.0 \
    psycopg2-binary==2.9.9

# Créer le même utilisateur que l'API (UID via build arg, défaut 1000) pour partager les permissions du volume
ARG APPUSER_UID=1000
RUN useradd -m -u ${APPUSER_UID} appuser || true

# Créer le répertoire artifacts et définir les permissions
RUN mkdir -p /mlflow/artifacts && \
    chown -R appuser:appuser /mlflow

# Passer à l'utilisateur non-root
USER appuser

EXPOSE 5000

# Le CMD est défini dans docker-compose.yml
