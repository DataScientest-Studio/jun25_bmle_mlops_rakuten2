import streamlit as st
import config


title = "Démonstration"
sidebar_name = "Démonstration"


def run():
    st.image(config.get_asset_path("presentation.svg"))

    st.title(title)
    
    