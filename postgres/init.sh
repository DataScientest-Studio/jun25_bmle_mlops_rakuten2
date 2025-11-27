#!/usr/bin/env bash
# Initialise la base Rakuten : schéma, ingestion et enregistrement du hash.
# Script conçu pour être exécuté directement dans le conteneur postgres.

set -euo pipefail

# ==============================================================================
# Constants
# ==============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_DIR="$ROOT_DIR/sql"

# ==============================================================================
# Functions
# ==============================================================================

initialize_db() {
  # Charger le hash des données depuis snapshot.json
  DATA_HASH=$(grep -o '"data_hash": *"[^"]*"' "$ROOT_DIR/snapshot.json" | sed -E 's/.*"data_hash": *"([^"]*)".*/\1/')

  if [[ -z "$DATA_HASH" ]]; then
    echo "Impossible de récupérer le hash des données."
    exit 1
  fi

  echo "Hash combiné : $DATA_HASH"

  SQL_FILES=(
    00_schema.sql
    01_tables.sql
    02_indexes.sql
    03_views.sql
    09_copy_raw.sql
    10_load_staging.sql
    11_upsert_items.sql
  )

  for sql_file in "${SQL_FILES[@]}"; do
    echo ">>> Exécution de $sql_file"
    psql -U mlops -d rakuten -v ON_ERROR_STOP=1 -f "$SQL_DIR/$sql_file"
  done

  echo ">>> Enregistrement du snapshot dans project.datasets"
  psql -U mlops -d rakuten -v ON_ERROR_STOP=1 -v data_hash="$DATA_HASH" -f "$SQL_DIR/12_register_snapshot.sql"

  echo ">>> Exécution des contrôles finaux"
  psql -U mlops -d rakuten -v ON_ERROR_STOP=1 -f "$SQL_DIR/13_checks.sql"

  echo ">>> Aperçu final"
  psql -U mlops -d rakuten -c "SELECT COUNT(*) AS nb_items FROM project.items;"
  psql -U mlops -d rakuten -c "SELECT prdtypecode, COUNT(*) FROM project.items GROUP BY 1 ORDER BY 2 DESC LIMIT 10;"
}

load_db_dump() {
  echo ">>> LOAD_DB_DUMP=1, chargement de rakuten_dump.sql au lieu de l'initialisation standard"
  psql -U mlops -d rakuten < "$ROOT_DIR/dump/rakuten_dump.sql"
}

create_additional_databases() {
  echo ">>> Création des bases Airflow et Mlflow"
  psql -U mlops -d postgres -c "CREATE DATABASE airflow" 2>/dev/null || true
  psql -U mlops -d postgres -c "CREATE DATABASE mlflow" 2>/dev/null || true
}

# ==============================================================================
# Main
# ==============================================================================

main() {
  echo "Postgres est prêt. Calcul du hash des CSV..."

  if [[ "${LOAD_DB_DUMP:-}" == "1" ]]; then
    load_db_dump
  else
    initialize_db
  fi

  create_additional_databases

  echo "Fin de l'initialisation."
}

main "$@"
