import streamlit as st
import config


title = "Architecture"
sidebar_name = "Architecture"


def run():
    st.image(config.get_asset_path("execution.svg"))

    st.title(title)
    
    tab1, tab2, tab3 = st.tabs(["Extract, transform and load", "Training", "Inference"])

    with tab1:
        st.image(config.get_asset_path("ETL.drawio.svg"))

    with tab2:
        st.image(config.get_asset_path("training.drawio.svg"))

    with tab3:
        st.image(config.get_asset_path("inference.drawio.svg"))