FROM python:3.11-slim

WORKDIR /mlflow

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Installation de MLflow et du driver PostgreSQL
RUN pip install --no-cache-dir \
    mlflow==3.6.0 \
    psycopg2-binary==2.9.9

EXPOSE 5000

# Le CMD est défini dans docker-compose.yml
