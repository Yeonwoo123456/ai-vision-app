
import base64
import io

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Image Test", layout="wide")
st.title("Image Rendering Test")

uploaded = st.file_uploader(
    "Upload a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        st.write("Image size:", image.size)

        st.html(
            f"""
            <div style="width:100%; overflow:hidden;">
                <img
                    src="data:image/png;base64,{encoded}"
                    alt="Uploaded image"
                    style="display:block; width:100%; height:auto;"
                />
            </div>
            """
        )

    except Exception as exc:
        st.error(f"Image display failed: {exc}")
