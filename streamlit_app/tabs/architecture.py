import streamlit as st
import config


title = "Architecture"
sidebar_name = "Architecture"


def run():
    st.image(config.get_asset_path("execution.svg"))

    st.title(title)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Extract, transform and load", "Training", "Inference", "Monitoring"])

    with tab1:
        st.image(config.get_asset_path("ETL.drawio.svg"))

    with tab2:
        st.image(config.get_asset_path("training.drawio.svg"))

    with tab3:
        st.image(config.get_asset_path("inference.drawio.svg"))
    
    with tab4:
        st.image(config.get_asset_path("monitoring.drawio.svg"))