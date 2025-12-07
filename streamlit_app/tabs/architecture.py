import streamlit as st
import config


title = "Architecture"
sidebar_name = "Architecture"


def run():
    st.image(config.get_asset_path("processing.svg"))

    st.title(title)
    
