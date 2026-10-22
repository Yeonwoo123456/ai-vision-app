"""UI and visual design helpers for AI Vision Lab.

Keep visual styling, page configuration, shared header/sidebar controls, and
image rendering here. Keep database/model logic in app.py.
"""

import base64
import io

import numpy as np
import streamlit as st
from PIL import Image


def configure_page():
    """Configure Streamlit page and apply global visual styling."""
    st.set_page_config(
        page_title="AI Vision Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1450px;
        }
        h1, h2, h3 {
            letter-spacing: -0.025em;
        }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.07);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 0.8rem 1rem;
            border-radius: 0.8rem;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 0.65rem;
            font-weight: 600;
            min-height: 2.6rem;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 0.8rem;
        }
        [data-testid="stExpander"] {
            border-radius: 0.8rem;
        }
        .vision-caption {
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.9rem;
            margin: 0.1rem 0 0.6rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the shared app header."""
    st.title("🔬 AI Vision Lab")
    st.caption("Facial Landmark Analysis · Profile Matching · AI Explanation")


def render_sidebar():
    """Render display controls and return their values to app.py."""
    with st.sidebar:
        st.header("⚙️ Settings")
        source = st.radio(
            "Image source",
            ["Camera", "Upload image"],
            key="ui_image_source",
        )
        st.divider()
        st.subheader("Visualization")
        show_landmarks = st.checkbox(
            "Show landmark visualization",
            value=True,
            key="ui_show_landmarks",
        )
        show_coordinates = st.checkbox(
            "Show landmark coordinates",
            value=False,
            key="ui_show_coordinates",
        )
        show_measurements = st.checkbox(
            "Show measurement lines on image",
            value=True,
            key="ui_show_measurements",
        )
        st.divider()
        st.caption("Images are processed in memory. Profile embeddings and user-entered profile details may be saved to the configured database.")
    return source, show_landmarks, show_coordinates, show_measurements


def show_image(image, caption):
    """Render an RGB image with a caption using a base64 data URL."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Invalid image dimensions.")
    if image.dtype != np.uint8:
        image = np.clip(np.nan_to_num(image), 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    st.markdown(f"**{caption}**")
    st.html(
        f"""
        <div style="width:100%;overflow:hidden;border-radius:0.8rem;">
          <img src="data:image/png;base64,{encoded}" alt="{caption}"
               style="display:block;width:100%;height:auto;object-fit:contain;" />
        </div>
        """
    )


def section_heading(title, description=None):
    """Render a consistent section heading and optional description."""
    st.subheader(title)
    if description:
        st.markdown(f'<div class="vision-caption">{description}</div>', unsafe_allow_html=True)
