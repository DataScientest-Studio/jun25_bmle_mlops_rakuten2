import streamlit as st
import config


title = "Modélisation"
sidebar_name = "Modélisation"


def run():
    st.image(config.get_asset_path("processing.svg"))

    st.title(title)
    
