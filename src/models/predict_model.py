"""
Module de prédiction pour les produits Rakuten.
===============================================

Ce module permet de charger un modèle entraîné et de faire des prédictions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProductPredictor:
    """
    Classe pour faire des prédictions sur les produits Rakuten.
    """

    def __init__(self, model_path: str):
        """Initialise le prédicteur en chargeant le modèle."""
        self.model_path = Path(model_path)
        self.model = None
        self.label_encoder = None
        self.load_model()
        logger.info(f"Prédicteur initialisé avec: {self.model_path}")

    def load_model(self) -> None:
        """Charge le modèle et le label encoder."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé: {self.model_path}"
            )

        logger.info(f"Chargement du modèle depuis: {self.model_path}")

        try:
            loaded = joblib.load(self.model_path)

            if isinstance(loaded, dict):
                if "model" not in loaded:
                    raise ValueError("Format de modèle invalide")

                self.model = loaded["model"]
                self.label_encoder = loaded.get("label_encoder")
                
                # Verify that the model is a Pipeline (full_pipeline) that can handle raw inputs
                from sklearn.pipeline import Pipeline
                if isinstance(self.model, Pipeline):
                    steps = self.model.steps
                    if len(steps) >= 2 and steps[0][0] == "features":
                        logger.info(
                            "Modèle détecté comme full_pipeline avec feature_pipeline. "
                            "Attend les inputs bruts: description, designation, image_binary"
                        )
                    elif len(steps) >= 1:
                        logger.info(
                            "Modèle détecté comme Pipeline avec %d step(s)", len(steps)
                        )

                if self.label_encoder is not None:
                    logger.info(
                        "Classes disponibles: %s",
                        getattr(self.label_encoder, "classes_", None),
                    )
            else:
                logger.warning("Ancien format de modèle détecté")
                self.model = loaded
                self.label_encoder = None

            logger.info("Modèle chargé: %s", type(self.model).__name__)

        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du modèle: {e}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fait des prédictions sur les données."""
        if self.model is None:
            raise ValueError("Modèle non chargé")

        # On laisse les DataFrame tels quels pour que le pipeline puisse
        # exploiter les colonnes (designation, description, etc.).
        if isinstance(X, pd.DataFrame):
            X_input = X
        else:
            X_input = np.asarray(X)
            if X_input.ndim != 2:
                raise ValueError(
                    f"X doit être 2D, forme actuelle: {X_input.shape}"
                )

        logger.info("Prédiction sur %d échantillons", len(X_input))

        y_pred_encoded = self.model.predict(X_input)

        if self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            logger.info("%d prédictions (décodées)", len(y_pred))
        else:
            y_pred = y_pred_encoded
            logger.info("%d prédictions (encodées)", len(y_pred))

        return y_pred

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Retourne les probabilités pour chaque classe."""
        if self.model is None:
            raise ValueError("Modèle non chargé")

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"{type(self.model).__name__} ne supporte pas predict_proba"
            )

        if isinstance(X, pd.DataFrame):
            X_input = X
        else:
            X_input = np.asarray(X)
            if X_input.ndim != 2:
                raise ValueError(
                    f"X doit être 2D, forme actuelle: {X_input.shape}"
                )

        logger.info("Prédiction de probabilités sur %d échantillons", len(X_input))
        probas = self.model.predict_proba(X_input)
        logger.info("Probabilités calculées: %s", probas.shape)

        return probas

    def predict_with_confidence(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """Prédictions avec confiance et top-3 classes probables."""
        if not hasattr(self.model, "predict_proba"):
            predictions = self.predict(X)
            return [
                {
                    "prediction": int(pred),
                    "confidence": None,
                    "message": "Modèle ne supporte pas predict_proba",
                }
                for pred in predictions
            ]

        predictions = self.predict(X)
        probas = self.predict_proba(X)

        results: List[Dict[str, Any]] = []
        for pred, proba in zip(predictions, probas):
            confidence = float(proba.max())
            top_3_idx = np.argsort(proba)[-3:][::-1]

            if self.label_encoder is not None:
                top_classes = [
                    {
                        "class": int(
                            self.label_encoder.inverse_transform([idx])[0]
                        ),
                        "probability": float(proba[idx]),
                    }
                    for idx in top_3_idx
                ]
            else:
                top_classes = [
                    {
                        "class": int(idx),
                        "probability": float(proba[idx]),
                    }
                    for idx in top_3_idx
                ]

            results.append(
                {
                    "prediction": int(pred),
                    "confidence": confidence,
                    "top_classes": top_classes,
                }
            )

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle."""
        info: Dict[str, Any] = {
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__ if self.model else None,
            "has_label_encoder": self.label_encoder is not None,
        }

        if self.label_encoder is not None:
            classes = getattr(self.label_encoder, "classes_", None)
            if classes is not None:
                info["num_classes"] = len(classes)
                info["classes"] = [int(c) for c in classes]

        if hasattr(self.model, "n_features_in_"):
            info["n_features"] = int(self.model.n_features_in_)

        return info

