FROM python:3.12-slim

# Dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (UID via build arg, défaut 1000)
ARG APPUSER_UID=1000
RUN useradd -m -u ${APPUSER_UID} appuser

WORKDIR /app

# Copier les dépendances Python
COPY api/requirements_working_exact.txt ./requirements_working_exact.txt

# Étape 1 : installation de PyTorch CPU (sans CUDA)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Étape 2 : filtrer torch/torchvision et tous les packages NVIDIA CUDA et installer le reste des dépendances
RUN grep -vE "^torch==|^torchvision==|^nvidia-|^triton==" requirements_working_exact.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt

# NLTK (stopwords, tokenisation)
RUN pip install --no-cache-dir nltk && \
    python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# Code applicatif
# Copier le répertoire src racine (pour src.utils, src.pipeline_steps, etc.)
COPY src/ ./src/
# Copier aussi models/ à la racine pour l'import models.predict_model
COPY src/models/ ./models/
# Copier le répertoire config (pour les fichiers de configuration)
COPY config/ ./config/
# Copier les fichiers API (main.py et schemas.py)
COPY api/src/main.py ./main.py
COPY api/src/schemas.py ./schemas.py

# Permissions
RUN mkdir -p models data/raw mlflow && \
    chmod -R 755 /usr/local/share/nltk_data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENV CUDA_VISIBLE_DEVICES=""
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "28800", "--timeout-graceful-shutdown", "30"]
