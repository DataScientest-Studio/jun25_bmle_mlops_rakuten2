import streamlit as st
import config


title = "Pistes d'amélioration"
sidebar_name = "Pistes d'amélioration"


def run():
    st.image(config.get_asset_path("processing.svg"))

    st.title(title)
    
    st.markdown("""
    - Améliorer le versionning des containers Docker via ajout d'un tag au build
    - Drift monitoring des modèles via Evidently et adaptation du pipeline de training en fonction du drift
    - Ajouter des tests unitaires pour les différentes étapes du pipeline
    - Ajouter de la CI/CD pour les différentes étapes du développement
    - Déploiement via Kubernetes avec une meilleure gestion des ressources/scaling
    - MAJ automatique des données via scraping des nouvelles données sur le site Rakuten
    """)