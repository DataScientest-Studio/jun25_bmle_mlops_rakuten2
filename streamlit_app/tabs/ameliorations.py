import streamlit as st
import config


title = "Pistes d'amélioration"
sidebar_name = "Pistes d'amélioration"


def run():
    st.image(config.get_asset_path("processing.svg"))

    st.title(title)
    
