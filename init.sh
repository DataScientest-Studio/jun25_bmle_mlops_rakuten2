#!/usr/bin/env bash
# demande bash sous Windows pour éviter la conversion des chemins par MSYS2
export MSYS_NO_PATHCONV=1

# Met à jour le suivi des données Rakuten
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
PY_SCRIPT="$ROOT_DIR/etl/manifest_and_hash.py"

echo "Postgres est prêt. Calcul du hash des CSV..."
DATA_HASH=$(python "$PY_SCRIPT")

if [[ -z "$DATA_HASH" ]]; then
  echo "Impossible de récupérer le hash des données."
  exit 1
fi

echo "Hash combiné : $DATA_HASH"Ò
