# pipelines/image_pipeline.py
"""
Pipeline images : charger depuis image_binary -> aplatir -> (réduire) -> normaliser

Mode unique: Utilise la colonne `image_binary` (bytes) au lieu de charger depuis fichiers.
Compatible avec les inputs de l'API: designation, description, image_binary.
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, Union, Iterable
import numpy as np
import logging

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, Normalizer
from sklearn.decomposition import PCA, TruncatedSVD
from src.features.image.loader import ImageLoader

from src.utils.profiling import profile_func

logger = logging.getLogger(__name__)


@profile_func
def _to_float32(X: np.ndarray) -> np.ndarray:
    """Convertir en float32."""
    return X.astype(np.float32, copy=False)


@profile_func
def _flatten_images(X: np.ndarray) -> np.ndarray:
    """Aplatir (n, H, W, C) -> (n, H*W*C)."""
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"Attendre un tenseur 4D (n,H,W,C) ; reçu shape={X.shape}")
    return X.reshape((X.shape[0], -1))


@profile_func
def _iter_pipeline_steps(obj: Any) -> Iterable[tuple[str, Any]]:
    """Itérer sur les (name, step) d'un objet type Pipeline/FeatureUnion."""
    if hasattr(obj, "steps"):
        for name, step in obj.steps:
            yield name, step
            yield from _iter_pipeline_steps(step)
    if hasattr(obj, "transformer_list"):
        for name, step in obj.transformer_list:
            yield name, step
            yield from _iter_pipeline_steps(step)
    for attr in ("estimator", "classifier", "regressor", "pipeline", "base_estimator"):
        if hasattr(obj, attr):
            inner = getattr(obj, attr)
            if inner is not obj:
                yield from _iter_pipeline_steps(inner)


@profile_func
def _find_reducer(obj: Any) -> Optional[tuple[str, Union[PCA, TruncatedSVD]]]:
    """Trouver la première étape PCA/TruncatedSVD."""
    for name, step in _iter_pipeline_steps(obj):
        if isinstance(step, (PCA, TruncatedSVD)):
            return name, step
    return None


@profile_func
def create_image_pipeline(
    image_size: Tuple[int, int] = (128, 128),
    dim_reduction: Optional[Dict[str, Any]] = None,
    memory: Optional[str] = None,
) -> SkPipeline:
    """
    Construit une pipeline image qui utilise image_binary (bytes).
    
    Étapes :
      1) Charger les images depuis image_binary (bytes)
      2) Convertir en float32
      3) Aplatir en vecteurs
      4) Réduire la dimension si demandé (PCA ou TruncatedSVD)
      5) Normaliser pour stabiliser l'entraînement

    Args:
        image_size: Taille cible (H, W)
        dim_reduction: Config de réduction dimensionnelle
        memory: Chemin de cache joblib (optionnel)

    Returns:
        Pipeline sklearn
    """
    cfg = dim_reduction or {}
    enabled = bool(cfg.get("enabled", False))
    method = str(cfg.get("method", "pca")).lower()
    n_comp = int(cfg.get("n_components", 150))
    rs = int(cfg.get("random_state", 42))

    steps = [
        ("loader", ImageLoader(image_size=image_size)),
        ("to_float", FunctionTransformer(_to_float32, accept_sparse=False)),
        ("flatten", FunctionTransformer(_flatten_images, accept_sparse=False)),
    ]

    if enabled:
        if method in ("svd", "truncated_svd"):
            steps += [
                ("svd", TruncatedSVD(n_components=n_comp, random_state=rs)),
                ("l2norm", Normalizer(copy=False)),
            ]
            logger.info("Réduction: TruncatedSVD (n_components=%d) + L2 norm", n_comp)
        elif method == "pca":
            steps += [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pca", PCA(
                    n_components=n_comp,
                    svd_solver="randomized",
                    whiten=True,
                    random_state=rs,
                )),
            ]
            logger.info("Réduction: PCA (n_components=%d, whiten=True)", n_comp)
        else:
            steps += [("scaler", StandardScaler(with_mean=False))]
            logger.warning("Méthode de réduction inconnue '%s' -> pas de réduction.", method)
    else:
        steps += [("scaler", StandardScaler(with_mean=False))]
        logger.info("Réduction désactivée -> standardiser.")

    return SkPipeline(steps=steps, memory=memory)


@profile_func
def create_image_pipeline_from_cfg(
    images_cfg: Dict[str, Any],
    memory: Optional[str] = None,
) -> SkPipeline:
    """
    Construit une pipeline image depuis la config TOML.

    Args:
        images_cfg: Dictionnaire `cfg["images"]`
        memory: Chemin de cache joblib (optionnel)

    Returns:
        Pipeline sklearn
    """
    size = tuple(images_cfg.get("size", [128, 128]))
    dr_cfg = images_cfg.get("dim_reduction", {}) or {}
    
    logger.info("Pipeline image créée (mode binary) avec size=%s", size)
    
    return create_image_pipeline(
        image_size=size,
        dim_reduction=dr_cfg,
        memory=memory,
    )


@profile_func
def diagnostic_reduction(pipe: SkPipeline) -> Dict[str, Any]:
    """
    Calculer des métriques de réduction après fit.
    """
    out: Dict[str, Any] = {}
    found = _find_reducer(pipe)
    if not found:
        out["reducer_type"] = None
        out["message"] = "Aucun réducteur trouvé dans la pipeline."
        return out

    name, reducer = found
    out["reducer_name"] = name
    out["reducer_type"] = type(reducer).__name__
    n_comp = getattr(reducer, "n_components_", None) or getattr(reducer, "n_components", None)
    out["n_components"] = int(n_comp) if n_comp is not None else None

    evr = getattr(reducer, "explained_variance_ratio_", None)
    if evr is not None:
        try:
            out["explained_variance_ratio_sum"] = float(np.sum(evr))
        except Exception:
            pass

    return out
