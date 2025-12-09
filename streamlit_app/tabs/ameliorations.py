import streamlit as st
import config


title = "Pistes d'amélioration"
sidebar_name = "Pistes d'amélioration"


def run():
    st.image(config.get_asset_path("processing.svg"))

    st.title(title)
    
    st.markdown("""
    - Améliorer le versionning des containers Docker via ajout d'un tag au build
    - Améliorer l'API d'inférence en permettant l'utilisation d'images hors du dataset
    - Ajouter des tests unitaires pour les différentes étapes du pipeline
    - Ajouter de la CI/CD pour les différentes étapes du pipeline
    """)