import os
from collections import OrderedDict

import streamlit as st

import config

from tabs import (
    intro,
    modelisation,
    architecture,
    demonstration,
    ameliorations,
    conclusion,
)


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open(os.path.join(os.path.dirname(__file__), "style.css"), "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (modelisation.sidebar_name, modelisation),
        (architecture.sidebar_name, architecture),
        (demonstration.sidebar_name, demonstration),
        (ameliorations.sidebar_name, ameliorations),
        (conclusion.sidebar_name, conclusion),
    ]
)


def run():
    # st.sidebar.image(
    #     "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
    #     width=200,
    # )
    st.sidebar.markdown("## Classification Multimodale des Produits - Rakuten France")

    tab_name = st.sidebar.radio("Navigation", list(TABS.keys()), 0, label_visibility="hidden")
    # st.sidebar.markdown("---")
    # st.sidebar.markdown(f"## {config.PROMOTION}")
    st.sidebar.markdown("## Auteurs")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
