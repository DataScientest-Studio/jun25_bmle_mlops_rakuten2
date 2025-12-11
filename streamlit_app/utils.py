import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import config
import base64

@st.cache_data
def load_data():
    """
    Load the selected data from the csv files
    """
    X_train = pd.read_csv(config.get_data_path("selected_X_train.csv"))
    y_train = pd.read_csv(config.get_data_path("selected_y_train.csv"))
    return X_train, y_train

def show_image_with_gray_border(path, caption=None):
    image = Image.open(path)
    bordered_image = ImageOps.expand(image, border=2, fill="gray")
    st.image(bordered_image, caption=caption)

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)