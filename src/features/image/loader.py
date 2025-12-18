# features/image_loader.py
"""
ImageLoader - Charge les images depuis des bytes (image_binary).

Mode unique: Utilise la colonne `image_binary` contenant des images en bytes bruts.
Compatible avec les inputs de l'API: designation, description, image_binary.
"""
from __future__ import annotations

import io
from typing import Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.profiling import profile_func, list_debug_add
import logging

logger = logging.getLogger(__name__)


class ImageLoader(BaseEstimator, TransformerMixin):
    """
    Charge les pixels d'images depuis des données binaires (bytes).
    
    Attend une colonne `image_binary` contenant des images en bytes bruts.
    Compatible avec les inputs de l'API FastAPI.
    
    - Redimensionne à `image_size` et normalise dans [0,1].
    - Renvoie un tenseur (n_samples, H, W, 3) en float32.
    - En cas d'erreur/image manquante, retourne un vecteur zéro (fallback).
    """

    @profile_func
    def __init__(
        self,
        image_size: Any = (128, 128),
        image_col: str = "image_binary",
        # Legacy parameters kept for sklearn.clone compatibility but ignored
        image_dir: Optional[str] = None,
        imgid_col: Optional[str] = None,
        pid_col: Optional[str] = None,
        ext: str = ".jpg",
    ):
        super().__init__()
        self.image_size = image_size
        self.image_col = image_col
        # Legacy params (ignored)
        self.image_dir = image_dir
        self.imgid_col = imgid_col
        self.pid_col = pid_col
        self.ext = ext

    @profile_func
    def fit(self, X=None, y=None):
        return self

    @profile_func
    def _resolve_size(self) -> Tuple[int, int]:
        """Convertit image_size en tuple (H, W)."""
        sz = self.image_size
        try:
            H = int(sz[0])
            W = int(sz[1])
        except Exception:
            H, W = 128, 128
        return H, W

    @profile_func
    def _process_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Traite des bytes d'image et retourne un array normalisé."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")
            
            H, W = self._resolve_size()
            img = img.resize((W, H))  # PIL: (width, height)
            arr = np.asarray(img, dtype=np.float32)
            
            # Normaliser en [0,1]
            if arr.max() > 1.0:
                arr /= 255.0
            
            return arr if arr.shape == (H, W, 3) else None
        except Exception:
            return None

    @profile_func
    def transform(self, X):
        """
        X: DataFrame avec colonne `image_binary` (bytes).
        Retour: np.ndarray (n, H, W, 3) float32 dans [0,1].
        """
        list_debug_add("ImageLoader.transform : " + str(X.shape[0]))
        H, W = self._resolve_size()
        n_samples = len(X)
        out = np.zeros((n_samples, H, W, 3), dtype=np.float32)

        if self.image_col not in X.columns:
            logger.warning(
                f"ImageLoader: Colonne '{self.image_col}' non trouvée. "
                f"Colonnes disponibles: {list(X.columns)}. "
                "Retourne des images vides."
            )
            return out

        image_binaries = X[self.image_col].tolist()
        
        for i, img_binary in enumerate(image_binaries):
            if pd.isna(img_binary) or img_binary is None:
                continue  # Garder les zéros (fallback)
            
            if isinstance(img_binary, bytes):
                arr = self._process_image_bytes(img_binary)
                if arr is not None:
                    out[i] = arr
            else:
                logger.warning(
                    f"ImageLoader: image_binary[{i}] n'est pas des bytes "
                    f"(type: {type(img_binary)}). Ignoré."
                )

        return out
