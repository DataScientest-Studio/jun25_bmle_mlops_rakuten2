"""
Étape 4 : Entraînement du modèle (Model Training).
==================================================

Cette étape entraîne le modèle sur les données transformées.

Responsabilités :
- Créer le modèle selon la configuration
- Entraîner sur les données transformées avec barre de progression
- Validation croisée avec affichage des folds
- Sauvegarder le modèle et le pipeline de features

Utilisation:
    from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline

    pipeline = ModelTrainingPipeline(config)
    model = pipeline.run(X_train_transformed, y_train_resampled)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.models.model_trainer import ModelTrainer
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Pipeline d'entraînement du modèle.

    Entraîne le modèle final sur les données transformées avec progression.

    Attributes:
        config: Configuration complète du projet
        trainer: Instance de ModelTrainer
        model: Modèle entraîné (après run)

    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = ModelTrainingPipeline(config)
        >>> model = pipeline.run(X_train_transformed, y_train_resampled)
    """

    def __init__(self, config: Any) -> None:
        """
        Initialise le pipeline d'entraînement.

        Args:
            config: Objet Config contenant tous les paramètres
        """
        self.config = config
        self.trainer: ModelTrainer | None = None
        self.model: Any | None = None

        logger.info("=" * 70)
        logger.info("ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE")
        logger.info("=" * 70)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """
        Entraîne le modèle sur les données.

        Args:
            X_train: Features transformées (n_samples, n_features)
            y_train: Labels (n_samples,)

        Returns:
            Modèle entraîné
        """
        logger.info("\n--- Entraînement du modèle ---")

        # Créer le trainer
        self.trainer = ModelTrainer(
            model_config=self.config.model,
            random_state=self.config.random_seed,
        )

        # Informations sur les données
        logger.info("Données d'entraînement: %s", X_train.shape)
        logger.info("Labels: %s", y_train.shape)
        logger.info("Nombre de classes: %d", len(np.unique(y_train)))

        # Type de matrice (sparse ou dense)
        if hasattr(X_train, "nnz"):
            logger.info("Matrice sparse - nnz: %d", X_train.nnz)
            density = X_train.nnz / np.prod(X_train.shape)
            logger.info("Densité: %.4f (%.2f%%)", density, density * 100)
        else:
            logger.info("Matrice dense")

        # ========================================
        # VALIDATION CROISÉE (si activée)
        # ========================================
        if self.config.get("cv.enabled", False):
            logger.info("\n--- Validation croisée ---")
            cv_scores = self.perform_cross_validation(X_train, y_train)

            # Stocker les scores pour y accéder depuis l'API
            self.trainer.cv_scores_ = {"f1_weighted": cv_scores}

            logger.info(
                "\nCV F1 (weighted) - Moyenne: %.4f (+/- %.4f)",
                cv_scores["mean"],
                cv_scores["std"],
            )
            logger.info("CV F1 (weighted) - Scores: %s", cv_scores["scores"])

        # ========================================
        # ENTRAÎNEMENT FINAL
        # ========================================
        model_name = self.config.model["name"].upper()
        with Timer(f"Entraînement {model_name}"):
            self.model = self.trainer.train(X_train, y_train)

        logger.info("[OK] Entraînement terminé")

        return self.model

    def perform_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Effectue une validation croisée stratifiée avec barre de progression.

        Args:
            X: Features
            y: Labels

        Returns:
            Dict avec scores, mean, std
        """
        # Paramètres CV depuis config
        n_splits = self.config.get("cv.splits", 3)
        shuffle = self.config.get("cv.shuffle", True)
        cv_random_state = self.config.get("cv.random_state", self.config.random_seed)
        show_progress = self.config.get("cv.show_progress", True)

        logger.info(
            "Paramètres CV: %d folds, shuffle=%s, random_state=%s",
            n_splits,
            shuffle,
            cv_random_state,
        )

        # Créer le StratifiedKFold
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=cv_random_state,
        )

        # Encoder y pour sklearn
        y_encoded = self.trainer.label_encoder.fit_transform(y)

        # Liste pour stocker les scores
        cv_scores: list[float] = []

        # ========================================
        # Boucle de CV avec progression
        # ========================================
        with Timer("Validation croisée"):
            # Créer l'itérateur avec ou sans tqdm
            if show_progress:
                fold_iterator = enumerate(
                    tqdm(
                        skf.split(X, y_encoded),
                        total=n_splits,
                        desc="Cross-validation",
                        unit="fold",
                        ncols=80,
                    ),
                    start=1,
                )
            else:
                fold_iterator = enumerate(skf.split(X, y_encoded), start=1)

            # Boucle sur les folds
            for fold_idx, (train_idx, val_idx) in fold_iterator:
                try:
                    logger.info("\n%s", "=" * 60)
                    logger.info("FOLD %d/%d", fold_idx, n_splits)
                    logger.info("%s", "=" * 60)
                    logger.info("Train: %d échantillons", len(train_idx))
                    logger.info("Val: %d échantillons", len(val_idx))

                    # Nettoyage mémoire avant le fold (sauf pour le premier)
                    if fold_idx > 1:
                        logger.debug("Nettoyage mémoire avant fold %d...", fold_idx)
                        gc.collect()

                    # Splitter les données
                    X_train_fold = X[train_idx]
                    y_train_fold = y_encoded[train_idx]
                    X_val_fold = X[val_idx]
                    y_val_fold = y_encoded[val_idx]

                    # Créer un modèle pour ce fold
                    fold_model = self.trainer.create_model()

                    # Entraîner
                    logger.info("Entraînement du fold...")

                    # ========================================
                    # EARLY STOPPING PAR FOLD
                    # ========================================
                    model_name = self.config.model["name"]

                    if model_name in ["xgb", "lgbm"]:
                        early_stopping_rounds = None
                        if model_name == "xgb":
                            early_stopping_rounds = self.config.model.get("xgb", {}).get(
                                "early_stopping_rounds",
                            )
                        elif model_name == "lgbm":
                            early_stopping_rounds = self.config.model.get("lgbm", {}).get(
                                "early_stopping_rounds",
                            )

                        if early_stopping_rounds:
                            logger.info(
                                "Early stopping activé pour ce fold (rounds=%s)",
                                early_stopping_rounds,
                            )

                            if model_name == "xgb":
                                fold_model.fit(
                                    X_train_fold,
                                    y_train_fold,
                                    eval_set=[(X_val_fold, y_val_fold)],
                                    verbose=False,
                                )
                                if hasattr(fold_model, "best_iteration"):
                                    logger.info(
                                        "Arrêt à l'itération %s",
                                        fold_model.best_iteration,
                                    )

                            elif model_name == "lgbm":
                                import lightgbm as lgb

                                fold_model.fit(
                                    X_train_fold,
                                    y_train_fold,
                                    eval_set=[(X_val_fold, y_val_fold)],
                                    callbacks=[
                                        lgb.early_stopping(
                                            stopping_rounds=early_stopping_rounds,
                                            verbose=False,
                                        ),
                                    ],
                                )
                                if hasattr(fold_model, "best_iteration_"):
                                    logger.info(
                                        "Arrêt à l'itération %s",
                                        fold_model.best_iteration_,
                                    )
                        else:
                            # Sans early stopping
                            fold_model.fit(X_train_fold, y_train_fold)
                    else:
                        # LR, SVC : pas d'early stopping
                        fold_model.fit(X_train_fold, y_train_fold)

                    # Prédire sur validation
                    y_pred = fold_model.predict(X_val_fold)

                    # Calculer F1
                    score = f1_score(y_val_fold, y_pred, average="weighted")
                    cv_scores.append(score)

                    logger.info("F1 Score (fold %d): %.4f", fold_idx, score)

                    # Nettoyage mémoire après le fold
                    del fold_model, y_pred
                    if hasattr(X_train_fold, "nnz"):  # Sparse matrix
                        # For sparse matrices, we can't easily delete, but we can clear references
                        pass
                    else:
                        del X_train_fold, X_val_fold
                    del y_train_fold, y_val_fold
                    
                    # Force garbage collection after each fold
                    gc.collect()
                    logger.debug("Mémoire nettoyée après fold %d", fold_idx)

                except MemoryError as e:
                    logger.error(
                        "Erreur mémoire lors du fold %d: %s", fold_idx, str(e)
                    )
                    logger.error(
                        "Recommandations: réduire num_leaves, max_bin, ou n_estimators"
                    )
                    raise
                except Exception as e:
                    logger.error(
                        "Erreur lors du fold %d: %s", fold_idx, str(e)
                    )
                    import traceback
                    logger.error(traceback.format_exc())
                    raise

        cv_scores_arr = np.array(cv_scores, dtype=float)

        return {
            "scores": [float(s) for s in cv_scores_arr],
            "mean": float(cv_scores_arr.mean()),
            "std": float(cv_scores_arr.std()),
        }

    def save_model(
        self,
        model: Any,
        output_path: str,
    ) -> None:
        """
        Sauvegarde le modèle.

        Args:
            model: Modèle entraîné
            output_path: Chemin de sauvegarde
        """
        logger.info("\n--- Sauvegarde du modèle ---")

        output_path_path = Path(output_path)
        output_path_path.parent.mkdir(parents=True, exist_ok=True)

        # Utilise la logique de ModelTrainer (model + label_encoder)
        self.trainer.save_model(model, str(output_path_path))

        logger.info("[OK] Modèle sauvegardé: %s", output_path_path)

    def save_full_pipeline(
        self,
        model: Any,
        feature_pipeline: Any,
        output_path: str,
    ) -> None:
        """
        Sauvegarde le pipeline complet (features + modèle).

        Utile pour la prédiction : on peut charger tout d'un coup.

        Args:
            model: Modèle entraîné
            feature_pipeline: Pipeline de features (sklearn)
            output_path: Chemin de sauvegarde
        """
        logger.info("\n--- Sauvegarde du pipeline complet ---")

        output_path_path = Path(output_path)
        output_path_path.parent.mkdir(parents=True, exist_ok=True)

        full_pipeline = {
            "feature_pipeline": feature_pipeline,
            "model": model,
            "label_encoder": self.trainer.label_encoder,
            "config": {
                "model_name": self.config.model["name"],
                "random_seed": self.config.random_seed,
            },
        }

        joblib.dump(full_pipeline, output_path_path)

        size_mb = output_path_path.stat().st_size / (1024 * 1024)
        logger.info("[OK] Pipeline complet sauvegardé: %s", output_path_path)
        logger.info("Taille: %.2f MB", size_mb)

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_pipeline: Any | None = None,
    ) -> Any:
        """
        Exécute le pipeline d'entraînement complet.

        Args:
            X_train: Features transformées
            y_train: Labels
            feature_pipeline: Pipeline de features (optionnel, pour sauvegarde complète)

        Returns:
            Modèle entraîné
        """
        with Timer("Entraînement du modèle"):
            # 1. Entraîner le modèle
            self.model = self.train_model(X_train, y_train)

            # 2. Sauvegarder le modèle seul avec timestamp versioning
            from datetime import datetime
            # Use timestamp from config if available (set by API), otherwise generate new one
            timestamp = self.config.get("training_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_path = self.config.paths.get(
                "model_out",
                "models/{kind}_{phase}_{timestamp}.joblib",
            )

            model_name = self.config.model["name"]
            model_path = model_path.replace("{kind}", model_name)
            model_path = model_path.replace("{phase}", "final")
            model_path = model_path.replace("{timestamp}", timestamp)

            self.save_model(self.model, model_path)

            # 3. Sauvegarder le pipeline complet (si fourni)
            full_pipeline_path = None
            if feature_pipeline is not None:
                full_pipeline_path = model_path.replace(
                    ".joblib",
                    "_full_pipeline.joblib",
                )
                self.save_full_pipeline(
                    self.model,
                    feature_pipeline,
                    full_pipeline_path,
                )

            # 4. Résumé final
            logger.info("\n%s", "=" * 70)
            logger.info("RÉSUMÉ DE L'ENTRAÎNEMENT")
            logger.info("%s", "=" * 70)
            logger.info("[OK] Modèle: %s", self.config.model["name"].upper())
            logger.info("[OK] Données: %s", X_train.shape)
            logger.info("[OK] Sauvegarde: %s", model_path)
            if full_pipeline_path is not None:
                logger.info("[OK] Pipeline complet: %s", full_pipeline_path)
            logger.info("%s\n", "=" * 70)

            return self.model


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    from src.pipeline_steps.stage03_data_transformation import (
        DataTransformationPipeline,
    )
    from src.utils.config import load_config
    from src.utils.logging_config import setup_logging

    setup_logging(level=logging.INFO)

    print("\n" + "=" * 70)
    print("Test de ModelTrainingPipeline")
    print("=" * 70 + "\n")

    try:
        # Charger la configuration
        cfg = load_config()

        # Stage 1: Ingestion
        stage1 = DataIngestionPipeline(cfg)
        X_tr, y_tr, X_te = stage1.run()

        # Stage 2: Validation
        stage2 = DataValidationPipeline(cfg)
        validation_ok = stage2.run(X_tr, y_tr, X_te)

        if not validation_ok:
            print("\n[ERROR] Validation échouée - arrêt du pipeline")
        else:
            # Stage 3: Transformation
            stage3 = DataTransformationPipeline(cfg)
            (
                X_tr_t,
                y_tr_t,
                X_te_t,
                feat_pipeline,
                feat_mapping,
            ) = stage3.run(X_tr, y_tr, X_te)

            # Stage 4: Training
            stage4 = ModelTrainingPipeline(cfg)
            mdl = stage4.run(X_tr_t, y_tr_t, feat_pipeline)

            print("\n[OK] Entraînement terminé avec succès !")
            print(f"  Modèle: {type(mdl).__name__}")
            print("  Sauvegarde dans: models/")

    except FileNotFoundError as exc:
        print(f"\n[ERROR] Erreur: {exc}")
        print("Assurez-vous que les fichiers CSV existent dans data/raw/")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n[ERROR] Erreur inattendue: {exc}")
        import traceback

        traceback.print_exc()





























