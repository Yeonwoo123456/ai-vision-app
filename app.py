
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="Image Test", layout="wide")
st.title("Image Rendering Test")

uploaded = st.file_uploader(
    "Upload a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image_bytes = uploaded.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.write("Image loaded successfully")
        st.write("Image size:", image.size)

        st.image(image, caption="Test image")

    except Exception as e:
        st.error(f"Image loading failed: {e}")
