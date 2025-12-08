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
        #### Résultats
        - Conclusion

        """
    )
