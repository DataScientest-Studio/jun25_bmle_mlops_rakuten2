import streamlit as st
import config


title = "Introduction"
sidebar_name = "Introduction"


def run():

    st.image(config.get_asset_path("shopping.svg"))

    st.title(title)

    st.markdown("---")

    st.header("Présentation du projet")
    st.write("""
    Ce projet a pour objectifs :
    - l'élaboration d'un modèle de classification multimodale (texte + image) de produits pour le site e-commerce Rakuten France
    - la mise en place d'une architecture MLOps pour le développement, l'entraînement et le déploiement du modèle
    - la mise en place d'une interface utilisateur pour l'utilisation du modèle
    """)

    st.header("En savoir plus")
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(config.get_asset_path("RIT_logo_big.jpg"))
    with cols[1]:
        st.markdown("""
        Pour consulter le détail du challenge et du jeu de données :  
        [Rakuten France Multimodal Product Data Classification — ENS Challenge](https://challengedata.ens.fr/challenges/35)
        """)
