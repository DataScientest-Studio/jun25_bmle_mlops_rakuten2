import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime
from mlflow import MlflowClient
import os

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rakuten_classification")

# Liste des modèles
models_to_register = [
    {
        "path": "models/model_full_pipeline.joblib",
        "name": "rakuten_full_pipeline",
        "description": "Pipeline complet avec preprocessing et modèle"
    },
    {
        "path": "models/model.joblib",
        "name": "rakuten_classifier",
        "description": "Modèle de classification Rakuten"
    },
    {
        "path": "models/lr_model_final.joblib",
        "name": "rakuten_logistic_regression",
        "description": "Logistic Regression finale"
    }
]

client = MlflowClient()

# Enregistrement de chaque modèle
for model_info in models_to_register:
    print(f"\n📦 Enregistrement de {model_info['name']}...")
    
    try:
        # Chargement du modèle
        model = joblib.load(model_info["path"])
        
        # Log le modèle dans MLflow (sans registered_model_name pour éviter l'erreur)
        with mlflow.start_run(run_name=f"{model_info['name']}_registration") as run:
            run_id = run.info.run_id
            
            # Log des paramètres
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("source", "existing_model")
            mlflow.log_param("registration_date", datetime.now().isoformat())
            
            # Sauvegarde temporaire du modèle
            temp_path = f"/tmp/{model_info['name']}.pkl"
            joblib.dump(model, temp_path)
            
            # Log comme artifact
            mlflow.log_artifact(temp_path, artifact_path="model")
            
            # Nettoyage
            os.remove(temp_path)
            
            print(f"✅ Run {run_id} créé")
        
        # Maintenant enregistrez dans le Registry
        print(f"🔄 Enregistrement dans le Registry...")
        
        # Créer le registered model
        try:
            client.create_registered_model(
                model_info["name"],
                description=model_info["description"]
            )
            print(f"✅ Registered model '{model_info['name']}' créé")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                print(f"ℹ️  Registered model '{model_info['name']}' existe déjà")
            else:
                raise
        
        # Créer une version
        model_source = f"runs:/{run_id}/model"
        mv = client.create_model_version(
            name=model_info["name"],
            source=model_source,
            run_id=run_id
        )
        print(f"✅ Version {mv.version} créée pour {model_info['name']}")
            
    except Exception as e:
        print(f"❌ Erreur pour {model_info['name']}: {e}")
        import traceback
        traceback.print_exc()

print("\n🎉 Enregistrement terminé ! Vérifiez http://localhost:5000")
