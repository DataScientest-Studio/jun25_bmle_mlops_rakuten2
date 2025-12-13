import streamlit as st
import pandas as pd
import requests
import os
import json
from PIL import Image as PILImage
import config
from pathlib import Path
import io

title = "Démonstration"
sidebar_name = "Démonstration"

# Configuration de l'API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict/"


def load_x_train_data():
    """Charge les données des produits depuis le CSV."""
    try:
        data_path = config.get_data_path("selected_X_train.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None


@st.cache_data
def load_y_train_data():
    """Charge les données des catégories depuis selected_y.csv."""
    try:
        y_path = config.get_data_path("selected_y_train.csv")
        df_y = pd.read_csv(y_path)
        return df_y
    except Exception as e:
        st.error(f"Erreur lors du chargement des catégories: {e}")
        return None


def call_predict_api(designation, description=None, image_file=None, api_url=None):
    """
    Appelle l'API FastAPI pour faire une prédiction avec multipart/form-data.
    
    Utilise l'endpoint /predict/ qui accepte les fichiers directement
    en bytes bruts (meilleures performances, pas de base64).
    """
    api_url = api_url or os.getenv("API_BASE_URL", "http://localhost:8000")
    predict_endpoint = f"{api_url}/predict/"
    
    if not image_file:
        return None, "Image requise"
    
    # Déterminer le nom du fichier et le type MIME
    if hasattr(image_file, 'name') and hasattr(image_file, 'type'):
        # Fichier uploadé via Streamlit (UploadedFile)
        filename = image_file.name
        content_type = image_file.type
    else:
        # BytesIO (depuis Form 2) - utiliser des valeurs par défaut
        filename = "image.jpg"
        content_type = "image/jpeg"
    
    # Préparer les données pour multipart/form-data avec (filename, file, content_type)
    files = {'image': (filename, image_file, content_type)}
    data = {'designation': designation}
    
    if description and pd.notna(description) and description.strip():
        data['description'] = description
    
    try:
        response = requests.post(predict_endpoint, files=files, data=data, timeout=30)
        
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


def load_product_image(imageid, productid):
    """Charge l'image d'un produit depuis le filesystem."""
    try:
        image_path = config.get_data_path(f"images/image_{imageid}_product_{productid}.jpg")
        if os.path.exists(image_path):
            return PILImage.open(image_path)
        # Essayer avec .png si .jpg n'existe pas
        image_path = config.get_data_path(f"images/image_{imageid}_product_{productid}.png")
        if os.path.exists(image_path):
            return PILImage.open(image_path)
        return None
    except Exception as e:
        return None


def display_prediction_result(result, error=None):
    """Affiche les résultats de prédiction dans une section partagée."""
    if error:
        st.error(f"❌ {error}")
        return
    
    if not result:
        st.warning("⚠️ Aucun résultat retourné par l'API.")
        return
    
    st.success("✅ Prédiction réussie!")
    
    # Afficher les résultats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_code = result.get("prediction")
        prediction_label = get_category_label(prediction_code)
        st.markdown("**🎯 Catégorie prédite**")
        st.markdown(f"<div style='font-size: 1.2em; padding: 0.5em 0; word-wrap: break-word;'>{prediction_label}</div>", unsafe_allow_html=True)
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


def run():
    st.title(title)
    
    st.markdown("""
    Cette page permet de tester l'API de prédiction de deux manières :
    - **Formulaire manuel** : Upload d'image et saisie manuelle des informations
    - **Sélection depuis les données** : Choix d'un produit existant dans les données d'entraînement
    
    **Note**: L'API utilise `multipart/form-data` pour une transmission optimale des images.
    """)
    
    # Initialiser session state pour stocker le dernier résultat
    if 'latest_prediction_result' not in st.session_state:
        st.session_state.latest_prediction_result = None
    if 'latest_prediction_error' not in st.session_state:
        st.session_state.latest_prediction_error = None
    
    # Charger les données d'entraînement
    df_x_train = load_x_train_data()
    df_y_train = load_y_train_data()
    
    # ============================================
    # FORMULAIRE 1 : Saisie manuelle
    # ============================================
    st.header("📝 Formulaire 1 : Saisie manuelle")
    
    with st.form("manual_input_form", clear_on_submit=False):
        st.subheader("📤 Upload d'image")
        uploaded_file = st.file_uploader(
            "Choisissez une image du produit",
            type=['jpg', 'jpeg', 'png'],
            help="Upload une image du produit à classifier (JPEG ou PNG)",
            key="manual_upload"
        )
        
        # Afficher l'image uploadée (largeur limitée à la moitié)
        if uploaded_file is not None:
            uploaded_file.seek(0)
            image = PILImage.open(uploaded_file)
            col_img, _ = st.columns(2)
            with col_img:
                st.image(image, caption="Image uploadée", use_container_width=True)
        
        st.subheader("📋 Informations du produit")
        
        # Inputs pour designation et description
        manual_designation = st.text_input(
            "Désignation du produit *",
            placeholder="Ex: iPhone 13 Pro Max 256GB",
            help="Titre/nom du produit",
            key="manual_designation"
        )
        
        manual_description = st.text_area(
            "Description du produit (optionnel)",
            placeholder="Description détaillée du produit...",
            help="Description longue du produit",
            key="manual_description"
        )
        
        # Bouton de prédiction pour le formulaire manuel
        manual_submit = st.form_submit_button(
            "🚀 Lancer la prédiction",
            type="primary",
            use_container_width=True
        )
        
        if manual_submit:
            if not manual_designation:
                st.session_state.latest_prediction_error = "❌ La désignation est requise"
            elif not uploaded_file:
                st.session_state.latest_prediction_error = "❌ Une image est requise"
            else:
                # Réinitialiser le pointeur du fichier pour l'envoyer
                uploaded_file.seek(0)
                
                with st.spinner("⏳ Appel de l'API en cours..."):
                    result, error = call_predict_api(
                        designation=manual_designation,
                        description=manual_description if manual_description and manual_description.strip() else None,
                        image_file=uploaded_file
                    )
                
                # Stocker le résultat le plus récent
                # Si succès, stocker le résultat et effacer l'erreur précédente
                # Si erreur, stocker l'erreur et effacer le résultat précédent
                if result:
                    st.session_state.latest_prediction_result = result
                    st.session_state.latest_prediction_error = None
                else:
                    st.session_state.latest_prediction_result = None
                    st.session_state.latest_prediction_error = error
    
    # ============================================
    # FORMULAIRE 2 : Sélection depuis les données existantes
    # ============================================
    st.header("🔍 Formulaire 2 : Sélection depuis les données d'entraînement")
    
    if df_x_train is not None and not df_x_train.empty and df_y_train is not None and not df_y_train.empty:
        # Dropdown HORS du formulaire pour mise à jour dynamique
        product_options = []
        for idx, row in df_x_train.iterrows():
            desig = str(row["designation"])
            product_options.append(desig)
        
        selected_index = st.selectbox(
            "Choisissez un produit:",
            range(len(product_options)),
            format_func=lambda x: product_options[x],
            key="product_selector"
        )
        
        # Afficher les informations du produit sélectionné (en lecture seule, HORS formulaire)
        if selected_index is not None:
            selected_row = df_x_train.iloc[selected_index]
            
            # Obtenir le prdtypecode depuis y_train
            prdtypecode_str = "Non disponible"
            if len(df_y_train) > selected_index:
                y_row = df_y_train.iloc[selected_index]
                
                # Si c'est un DataFrame, extraire la colonne prdtypecode ou la première colonne
                if isinstance(y_row, pd.Series):
                    if 'prdtypecode' in y_row.index:
                        prdtypecode = y_row['prdtypecode']
                    else:
                        prdtypecode = y_row.iloc[0] if len(y_row) > 0 else None
                elif isinstance(y_row, pd.DataFrame):
                    if 'prdtypecode' in y_row.columns:
                        prdtypecode = y_row['prdtypecode'].iloc[0]
                    else:
                        prdtypecode = y_row.iloc[0, 0]
                else:
                    prdtypecode = y_row
                
                prdtypecode_str = get_category_label(prdtypecode)
            
            # Afficher les informations en lecture seule
            st.subheader("📦 Informations du produit sélectionné")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Afficher l'image du produit
                if 'imageid' in selected_row and 'productid' in selected_row:
                    product_image = load_product_image(
                        selected_row['imageid'], 
                        selected_row['productid']
                    )
                    if product_image:
                        st.image(product_image, caption="Image du produit", use_container_width=True)
                    else:
                        st.warning("⚠️ Image non trouvée sur le filesystem")
                else:
                    st.warning("⚠️ imageid ou productid manquant dans les données")
            
            with col2:
                # Afficher designation (lecture seule)
                st.markdown(f"**Désignation :**")
                st.text(selected_row["designation"])
                
                # Afficher description (lecture seule)
                if pd.notna(selected_row.get("description")):
                    st.markdown(f"**Description :**")
                    st.markdown(selected_row.get("description"), unsafe_allow_html=True)
                else:
                    st.markdown(f"**Description:** *Non disponible*")
                
                # Afficher prdtypecode_str (lecture seule)
                st.markdown(f"**Catégorie :**")
                st.text(prdtypecode_str)
            
            # Bouton de prédiction (hors formulaire, utilise st.button)
            if st.button(
                "🚀 Lancer la prédiction",
                type="primary",
                use_container_width=True,
                key="existing_predict_button"
            ):
                # Charger l'image du produit sélectionné
                if 'imageid' in selected_row and 'productid' in selected_row:
                    product_image = load_product_image(
                        selected_row['imageid'], 
                        selected_row['productid']
                    )
                    if product_image:
                        # Convertir l'image en bytes pour l'API
                        img_bytes = io.BytesIO()
                        product_image.save(img_bytes, format='JPEG')
                        img_bytes.seek(0)
                        
                        with st.spinner("⏳ Appel de l'API en cours..."):
                            result, error = call_predict_api(
                                designation=selected_row["designation"],
                                description=selected_row.get("description") if pd.notna(selected_row.get("description")) else None,
                                image_file=img_bytes
                            )
                        
                        # Stocker le résultat le plus récent
                        st.session_state.latest_prediction_result = result
                        st.session_state.latest_prediction_error = error
                    else:
                        st.session_state.latest_prediction_error = "❌ Image non trouvée sur le filesystem"
                else:
                    st.session_state.latest_prediction_error = "❌ imageid ou productid manquant dans les données"
    else:
        st.warning("⚠️ Les données d'entraînement ne sont pas disponibles.")
    
    # ============================================
    # SECTION PARTAGÉE : Résultats de prédiction
    # ============================================
    st.header("🔮 Résultats de prédiction")
    st.markdown("**Cette section affiche toujours le résultat de la prédiction la plus récente.**")
    
    # Afficher le dernier résultat stocké
    if st.session_state.latest_prediction_result is not None or st.session_state.latest_prediction_error is not None:
        display_prediction_result(
            st.session_state.latest_prediction_result,
            st.session_state.latest_prediction_error
        )
    else:
        st.info("ℹ️ Aucune prédiction n'a encore été effectuée. Utilisez l'un des formulaires ci-dessus pour lancer une prédiction.")
    