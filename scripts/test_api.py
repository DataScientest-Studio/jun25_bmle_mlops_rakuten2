"""
Script de test pour l'API Rakuten MLOps.
========================================

Ce script teste tous les endpoints de l'API.

Usage:
    python scripts/test_api.py
    python scripts/test_api.py --train  # Inclut test d'entraînement
"""
import requests
import json
import time
import sys


API_URL = "http://localhost:8000"


def print_section(title: str):
    """Affiche un titre de section."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def test_health_check():
    """Test du health check."""
    print_section("1. Test Health Check")
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("✅ Health check réussi")
            return True
        else:
            print("❌ Health check échoué")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_training(model_name: str = "lr"):
    """Test de l'entraînement d'un modèle."""
    print_section(f"2. Test Training - Modèle {model_name.upper()}")
    
    payload = {
        "model_name": model_name,
        "experiment_name": "test_api",
        "run_name": f"test_{model_name}_{int(time.time())}"
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\n⏳ Entraînement en cours... (cela peut prendre quelques minutes)")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/training/",
            json=payload,
            timeout=600  # 10 minutes max
        )
        duration = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        print(f"\n⏱️ Durée: {duration:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("✅ Entraînement réussi")
                return True
        
        print("❌ Entraînement échoué")
        return False
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_model_info():
    """Test de l'endpoint model/info."""
    print_section("3. Test Model Info")
    
    try:
        response = requests.get(f"{API_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code in [200, 503]:
            print("✅ Endpoint model/info fonctionne")
            return True
        else:
            print("❌ Erreur")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def run_all_tests(train_model: bool = False):
    """Lance tous les tests."""
    print("\n" + "🚀" * 35)
    print("   Tests de l'API Rakuten MLOps")
    print("🚀" * 35)
    
    results = {}
    
    # Test 1: Health check
    results["health"] = test_health_check()
    time.sleep(1)
    
    # Test 2: Model info
    results["model_info"] = test_model_info()
    time.sleep(1)
    
    # Test 3: Training (optionnel)
    if train_model:
        print("\n⚠️ ATTENTION: L'entraînement va prendre plusieurs minutes...")
        results["training"] = test_training("lr")
    else:
        print("\n⏭️ Entraînement ignoré (utilisez --train pour inclure)")
        results["training"] = None
    
    # Résumé
    print_section("RÉSUMÉ DES TESTS")
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}")
        elif result is False:
            print(f"❌ {test_name}")
        else:
            print(f"⏭️ {test_name} (ignoré)")
    
    print(f"\n📊 Résultats: {passed} réussis, {failed} échoués, {skipped} ignorés")


if __name__ == "__main__":
    # Vérifier si l'API est accessible
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✅ API accessible à {API_URL}")
    except:
        print(f"❌ API non accessible à {API_URL}")
        print("Assurez-vous que l'API est lancée avec: uvicorn api.main:app --reload")
        sys.exit(1)
    
    # Option pour entraîner un modèle
    train = "--train" in sys.argv or "-t" in sys.argv
    
    # Lancer les tests
    run_all_tests(train_model=train)
