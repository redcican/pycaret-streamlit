import base64
import streamlit as st
from PIL import Image
from pathlib import Path


def load_nav_image(image_path):
    image_eido = Image.open(image_path)
    st.sidebar.image(image_eido, use_column_width=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def load_header_image(image_path):
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes(image_path))
    st.markdown(
        header_html, unsafe_allow_html=True,
    )
    st.markdown("---")