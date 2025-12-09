import streamlit as st
import config


title = "Conclusion"
sidebar_name = "Conclusion"


def run():
    st.image(config.get_asset_path("team.svg"))

    st.title(title)
    st.markdown("---")
    st.markdown(
        """
        Nous avons pu mettre en place un pipeline de machine learning complet, de l'extraction des données à la prédiction.
        
        Ce projet a mis en avant les différents besoins de chaque métier concerné par le projet : data scientist, data engineer, mlops, etc.
        
        De nombreuses améliorations possibles ont été identifiées et nous sommes conscients que le marché évolue constamment et rapidement : il est donc essentiel de rester à jour et de suivre les dernières tendances.
        """
    )
