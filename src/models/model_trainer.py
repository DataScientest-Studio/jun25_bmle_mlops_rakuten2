"""
Module d'entraînement des modèles - Classe ModelTrainer.
=======================================================

Ce module implémente la classe ModelTrainer qui gère l'entraînement
des différents modèles (LR, SVC, XGB, LGBM) avec leurs hyperparamètres.

Utilisation:
    from src.models.model_trainer import ModelTrainer

    trainer = ModelTrainer(config)
    model = trainer.train(X_train, y_train)
    trainer.save_model(model, "models/best_model.joblib")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Classe pour gérer l'entraînement des modèles de classification.

    Cette classe encapsule la logique d'entraînement et de sauvegarde
    des modèles selon la configuration fournie.

    Attributes:
        config: Configuration du modèle (dict depuis config.toml)
        random_state: Graine aléatoire pour la reproductibilité
        model: Modèle entraîné (None avant le training)

    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> trainer = ModelTrainer(config.model, random_state=42)
        >>> model = trainer.train(X_train, y_train)
        >>> trainer.save_model(model, "models/my_model.joblib")
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        random_state: int = 42,
    ) -> None:
        """
        Initialise le ModelTrainer.

        Args:
            model_config: Configuration du modèle (section [model] du TOML)
            random_state: Graine aléatoire (par défaut 42)
        """
        self.config = model_config
        self.random_state = random_state
        self.model: Optional[Any] = None
        # Pour gérer des classes non-séquentielles (10, 40, 50, ...)
        self.label_encoder = LabelEncoder()

        logger.info("ModelTrainer initialisé avec modèle: %s", self.config["name"])

    def create_model(self) -> Any:
        """
        Crée un modèle selon la configuration.

        Returns:
            Modèle sklearn non entraîné

        Raises:
            ValueError: Si le nom du modèle est inconnu
        """
        model_name = self.config["name"].lower()
        logger.info("Création du modèle: %s", model_name)

        # Logistic Regression
        if model_name == "lr":
            lr_params = self.config.get("lr", {})
            model = LogisticRegression(
                random_state=self.random_state,
                **lr_params,
            )
            logger.info("  Solver: %s", lr_params.get("solver", "default"))
            logger.info("  Penalty: %s", lr_params.get("penalty", "default"))
            logger.info("  C: %s", lr_params.get("C", 1.0))

        # Linear SVC
        elif model_name == "svc":
            svc_params = self.config.get("svc", {})
            model = LinearSVC(
                random_state=self.random_state,
                **svc_params,
            )
            logger.info("  C: %s", svc_params.get("C", 1.0))
            logger.info("  Loss: %s", svc_params.get("loss", "squared_hinge"))

        # XGBoost
        elif model_name == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "XGBoost non installé. Installez avec: pip install xgboost"
                ) from exc

            xgb_params = self.config.get("xgb", {})
            model = XGBClassifier(
                random_state=self.random_state,
                **xgb_params,
            )
            logger.info("  N estimators: %s", xgb_params.get("n_estimators", 100))
            logger.info("  Learning rate: %s", xgb_params.get("learning_rate", 0.1))

        # LightGBM
        elif model_name == "lgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError(
                    "LightGBM non installé. Installez avec: pip install lightgbm"
                ) from exc

            lgbm_params = self.config.get("lgbm", {})
            model = LGBMClassifier(
                random_state=self.random_state,
                **lgbm_params,
            )
            logger.info("  N estimators: %s", lgbm_params.get("n_estimators", 100))
            logger.info("  Learning rate: %s", lgbm_params.get("learning_rate", 0.1))
            logger.info("  Num leaves: %s", lgbm_params.get("num_leaves", 31))
            logger.info("  Max bin: %s", lgbm_params.get("max_bin", 255))
            num_threads = lgbm_params.get("num_threads", "auto")
            logger.info("  Num threads: %s", num_threads)

        else:
            raise ValueError(
                f"Modèle inconnu: {model_name}. "
                "Modèles supportés: 'lr', 'svc', 'xgb', 'lgbm'",
            )

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> Any:
        """
        Entraîne le modèle sur les données fournies avec early stopping optionnel.

        Args:
            X_train: Features d'entraînement (n_samples, n_features)
            y_train: Labels d'entraînement (n_samples,)
            X_val: Features de validation (optionnel, pour early stopping)
            y_val: Labels de validation (optionnel, pour early stopping)
            **fit_params: Paramètres supplémentaires pour fit()

        Returns:
            Modèle entraîné
        """
        with Timer(f"Entraînement du modèle {self.config['name']}"):
            # Créer le modèle
            self.model = self.create_model()

            # Informations sur les données
            logger.info(
                "Dimensions d'entraînement: X=%s, y=%s",
                X_train.shape,
                y_train.shape,
            )

            # Encodage des labels (10, 40, 50... -> 0, 1, 2...)
            unique_classes = np.unique(y_train)
            logger.info("Classes originales: %s", sorted(unique_classes))

            y_train_encoded = self.label_encoder.fit_transform(y_train)

            logger.info("Classes après encodage: %s", np.unique(y_train_encoded))
            logger.info("Nombre de classes: %s", len(unique_classes))

            # Early stopping pour XGB et LGBM
            model_name = self.config["name"].lower()

            if model_name in ["xgb", "lgbm"]:
                early_stopping_rounds: Optional[int] = None
                if model_name == "xgb":
                    early_stopping_rounds = self.config.get("xgb", {}).get(
                        "early_stopping_rounds",
                    )
                elif model_name == "lgbm":
                    early_stopping_rounds = self.config.get("lgbm", {}).get(
                        "early_stopping_rounds",
                    )

                if early_stopping_rounds:
                    # Création éventuelle d'un split de validation
                    if X_val is None or y_val is None:
                        from sklearn.model_selection import train_test_split

                        logger.info(
                            "Early stopping activé (rounds=%s)", early_stopping_rounds
                        )
                        logger.info(
                            "Création d'un split validation (20%% des données)",
                        )

                        (
                            X_train_split,
                            X_val_split,
                            y_train_split,
                            y_val_split,
                        ) = train_test_split(
                            X_train,
                            y_train_encoded,
                            test_size=0.2,
                            random_state=self.random_state,
                            stratify=y_train_encoded,
                        )
                    else:
                        logger.info(
                            "Early stopping activé (rounds=%s)", early_stopping_rounds
                        )
                        X_train_split = X_train
                        y_train_split = y_train_encoded
                        X_val_split = X_val
                        y_val_split = self.label_encoder.transform(y_val)

                    # XGBoost
                    if model_name == "xgb":
                        self.model.fit(
                            X_train_split,
                            y_train_split,
                            eval_set=[(X_val_split, y_val_split)],
                            verbose=50,  # log toutes les 50 itérations
                            **fit_params,
                        )

                        best_iteration = getattr(self.model, "best_iteration", None)
                        n_estimators = self.config.get("xgb", {}).get(
                            "n_estimators",
                            "unknown",
                        )
                        logger.info(
                            "Early stopping à l'itération %s / %s",
                            best_iteration,
                            n_estimators,
                        )

                    # LightGBM
                    elif model_name == "lgbm":
                        import lightgbm as lgb

                        self.model.fit(
                            X_train_split,
                            y_train_split,
                            eval_set=[(X_val_split, y_val_split)],
                            callbacks=[
                                lgb.early_stopping(
                                    stopping_rounds=early_stopping_rounds,
                                    verbose=False,
                                ),
                                lgb.log_evaluation(period=50),
                            ],
                            **fit_params,
                        )

                        best_iteration = getattr(self.model, "best_iteration_", None)
                        n_estimators = self.config.get("lgbm", {}).get(
                            "n_estimators",
                            "unknown",
                        )
                        logger.info(
                            "Early stopping à l'itération %s / %s",
                            best_iteration,
                            n_estimators,
                        )
                else:
                    # Pas d'early stopping
                    self.model.fit(X_train, y_train_encoded, **fit_params)

            else:
                # LR, SVC : pas d'early stopping
                self.model.fit(X_train, y_train_encoded, **fit_params)

            logger.info("Entraînement terminé")

        return self.model

    def save_model(
        self,
        model: Any,
        output_path: str,
    ) -> None:
        """
        Sauvegarde le modèle dans un fichier .joblib.

        Args:
            model: Modèle entraîné à sauvegarder
            output_path: Chemin de sauvegarde (ex: "models/model.joblib")
        """
        output_path_path = Path(output_path)
        output_path_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Sauvegarde du modèle dans: %s", output_path_path)

        # Sauvegarder le modèle et le label_encoder ensemble
        model_bundle = {
            "model": model,
            "label_encoder": self.label_encoder,
        }
        joblib.dump(model_bundle, output_path_path)

        size_mb = output_path_path.stat().st_size / (1024 * 1024)
        logger.info("Modèle sauvegardé (%.2f MB)", size_mb)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions et les décode automatiquement.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Prédictions dans les classes originales (10, 40, 50, etc.)
        """
        if self.model is None:
            raise ValueError(
                "Le modèle n'est pas entraîné. Appelez train() avant predict().",
            )

        # Prédire avec labels encodés (0, 1, 2, ...)
        y_pred_encoded = self.model.predict(X)

        # Décoder vers les classes originales (10, 40, 50, ...)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred

    @staticmethod
    def load_model(model_path: str) -> Any:
        """
        Charge un modèle depuis un fichier .joblib.

        Args:
            model_path: Chemin vers le fichier .joblib

        Returns:
            Tuple (model, label_encoder) ou juste model si ancien format.
        """
        model_path_path = Path(model_path)

        if not model_path_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path_path}")

        logger.info("Chargement du modèle depuis: %s", model_path_path)
        loaded = joblib.load(model_path_path)

        # dict avec model + label_encoder
        if isinstance(loaded, dict) and "model" in loaded:
            logger.info("Modèle et LabelEncoder chargés")
            return loaded

        # Ancien format: juste le modèle
        logger.warning("Ancien format détecté (sans LabelEncoder)")
        return loaded

