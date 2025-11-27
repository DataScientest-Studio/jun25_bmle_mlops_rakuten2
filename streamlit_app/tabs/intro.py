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
    Ce projet consiste en l'élaboration d'un modèle robuste pour effectuer de la classification automatique de produits issus du catalogue du site e-commerce Rakuten France, 
    utilisant à la fois les données textuelles (titre et description) et les images fournies pour chaque article.
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
