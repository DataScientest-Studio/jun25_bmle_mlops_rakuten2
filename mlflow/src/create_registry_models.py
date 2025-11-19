import mlflow
from mlflow.tracking import MlflowClient

# URI MLflow (adapter si besoin)
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

models_to_register = [
    {
        "name": "rakuten_full_pipeline",
        "description": "Pipeline complet avec preprocessing et modèle",
    },
    {
        "name": "rakuten_classifier",
        "description": "Modèle de classification Rakuten",
    },
    {
        "name": "rakuten_logistic_regression",
        "description": "Logistic Regression finale",
    },
]

for model_info in models_to_register:
    name = model_info["name"]
    description = model_info["description"]

    print(f"\nCréation du modèle {name}...")
    try:
        client.create_registered_model(name, description=description)
        print(f"Registered model '{name}' créé.")
    except Exception as e:
        if "ALREADY_EXISTS" in str(e):
            print(f"Registered model '{name}' existe déjà.")
        else:
            print(f"Erreur: {e}")

print("\nTerminé. Vérifiez http://localhost:5000 dans l'onglet Models.")
