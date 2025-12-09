import streamlit as st
import pandas as pd
import requests
import os
import json
from pathlib import Path
import config

title = "Démonstration"
sidebar_name = "Démonstration"

# Configuration de l'API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict/"


def load_product_data():
    """Charge les données des produits depuis le CSV."""
    try:
        data_path = config.get_data_path("selected_X_train.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None


def get_image_path(imageid):
    """Retourne le chemin de l'image si elle existe."""
    if pd.isna(imageid):
        return None
    
    # Chercher l'image dans le dossier data/images
    images_dir = Path(config.DATA_DIR) / "images"
    image_filename = f"image_{int(imageid)}_product_*.jpg"
    
    # Chercher tous les fichiers correspondants
    matching_files = list(images_dir.glob(f"image_{int(imageid)}_product_*.jpg"))
    
    if matching_files:
        return str(matching_files[0])
    return None


def call_predict_api(designation, description=None, productid=None, imageid=None, api_url=None):
    """Appelle l'API FastAPI pour faire une prédiction."""
    # Lire la variable d'environnement dynamiquement à chaque appel
    api_url = api_url or os.getenv("API_BASE_URL", "http://localhost:8000")
    
    predict_endpoint = f"{api_url}/predict/"
    
    payload = {
        "designation": designation,
    }
    
    if description and pd.notna(description) and description.strip():
        payload["description"] = description
    
    if productid and pd.notna(productid):
        payload["productid"] = int(productid)
    
    if imageid and pd.notna(imageid):
        payload["imageid"] = int(imageid)
    
    try:
        response = requests.post(predict_endpoint, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 501:
            return None, "L'endpoint de prédiction n'est pas encore implémenté sur le serveur."
        else:
            error_detail = response.json().get("detail", "Erreur inconnue")
            return None, f"Erreur API ({response.status_code}): {error_detail}"
    except requests.exceptions.ConnectionError:
        return None, f"Impossible de se connecter à l'API à l'adresse {api_url}. Vérifiez que le serveur FastAPI est démarré."
    except requests.exceptions.Timeout:
        return None, "Timeout: L'API a pris trop de temps à répondre."
    except Exception as e:
        return None, f"Erreur lors de l'appel API: {str(e)}"


@st.cache_data
def load_labels_map():
    """Charge le mapping des labels depuis labels_map.json (mis en cache)."""
    labels_map_path = config.get_asset_path("labels_map.json")
    with open(labels_map_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_category_label(prdtypecode):
    """Retourne le label de la catégorie depuis labels_map.json."""
    try:
        labels_map = load_labels_map()
        
        # Convertir le prdtypecode en string pour la recherche dans le JSON
        if prdtypecode is None or (hasattr(pd, 'isna') and pd.isna(prdtypecode)):
            return f"Catégorie inconnue"
        
        prdtypecode_str = str(int(prdtypecode))
        
        if prdtypecode_str in labels_map:
            return labels_map[prdtypecode_str]
        return f"Catégorie {prdtypecode}"
    except (ValueError, TypeError) as e:
        # Si la conversion en int échoue
        return f"Catégorie {prdtypecode}"
    except Exception as e:
        return f"Catégorie {prdtypecode}"


def run():
    st.title(title)
    
    st.markdown("""
    Cette page permet de tester l'API de prédiction en sélectionnant un produit 
    depuis les données d'entraînement et en affichant la prédiction du modèle.
    """)
    
    # Charger les données
    df = load_product_data()
    
    if df is None or df.empty:
        st.error("Aucune donnée disponible.")
        return
    
    # Créer un dropdown pour sélectionner un produit
    st.subheader("📦 Sélection d'un produit")
    
    # Créer une liste de produits pour le dropdown (designation + productid)
    product_options = []
    for idx, row in df.iterrows():
        designation = str(row["designation"])[:80] + "..." if len(str(row["designation"])) > 80 else str(row["designation"])
        productid = row.get("productid", idx)
        product_options.append(f"{designation} (ID: {productid})")
    
    selected_index = st.selectbox(
        "Choisissez un produit:",
        range(len(product_options)),
        format_func=lambda x: product_options[x],
        key="product_selector"
    )
    
    if selected_index is not None:
        selected_row = df.iloc[selected_index]
        
        # Afficher les informations du produit sélectionné
        st.subheader("📋 Informations du produit")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Désignation:** {selected_row['designation']}")
            st.markdown(f"**Product ID:** {selected_row.get('productid', 'N/A')}")
            st.markdown(f"**Image ID:** {selected_row.get('imageid', 'N/A')}")
            if pd.notna(selected_row.get("description")) and str(selected_row.get("description")).strip():
                with st.expander("📝 Description"):
                    st.text(selected_row["description"])
            else:
                st.info("ℹ️ Aucune description disponible pour ce produit.")

        
        with col2:
            # Afficher l'image si disponible
            imageid = selected_row.get("imageid")
            if pd.notna(imageid):
                image_path = get_image_path(imageid)
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption=f"Image du produit (ID: {int(imageid)})", use_container_width=True)
                else:
                    st.info("🖼️ Image non trouvée")
        
        # Bouton pour lancer la prédiction
        st.subheader("🔮 Prédiction")
        
        if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
            with st.spinner("⏳ Appel de l'API en cours..."):
                result, error = call_predict_api(
                    designation=selected_row["designation"],
                    description=selected_row.get("description"),
                    productid=selected_row.get("productid"),
                    imageid=selected_row.get("imageid")
                )
            
            if error:
                st.error(f"❌ {error}")
            elif result:
                st.success("✅ Prédiction réussie!")
                
                # Afficher les résultats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction_code = result.get("prediction")
                    prediction_label = get_category_label(prediction_code)
                    st.metric("🎯 Catégorie prédite", prediction_label)
                    st.caption(f"Code: {prediction_code}")
                
                with col2:
                    confidence = result.get("confidence")
                    if confidence is not None:
                        st.metric("📊 Confiance", f"{confidence:.2%}")
                    else:
                        st.info("Confiance non disponible")
                
                with col3:
                    top_classes = result.get("top_classes")
                    if top_classes:
                        st.metric("🏆 Top classes", len(top_classes))
                
                # Afficher les top classes si disponibles
                if result.get("top_classes"):
                    st.subheader("🏆 Top classes prédites")
                    top_classes_data = []
                    for i, class_info in enumerate(result["top_classes"], 1):
                        class_code = class_info.get("class") or class_info.get("prdtypecode")
                        class_prob = class_info.get("probability") or class_info.get("confidence", 0)
                        class_label = get_category_label(class_code)
                        top_classes_data.append({
                            "Rang": i,
                            "Catégorie": class_label,
                            "Code": class_code,
                            "Probabilité": f"{class_prob:.2%}"
                        })
                    
                    st.dataframe(
                        pd.DataFrame(top_classes_data),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Afficher la réponse JSON brute dans un expander
                with st.expander("📄 Réponse JSON complète"):
                    st.json(result)
            else:
                st.warning("⚠️ Aucun résultat retourné par l'API.")
    