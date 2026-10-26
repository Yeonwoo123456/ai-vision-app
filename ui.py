import base64
import html
import io

import numpy as np
import streamlit as st
from PIL import Image

def configure_page():
    st.set_page_config(
        page_title="AI Vision Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --vision-ink: #15243a;
            --vision-muted: #68778b;
            --vision-line: #dfe7f1;
            --vision-blue: #315efb;
            --vision-canvas: #f5f7fb;
        }
        .stApp {
            background: radial-gradient(ellipse at 10% 0%, rgba(49,94,251,.065), transparent 32rem), var(--vision-canvas);
        }
        [data-testid="stHeader"] { background: rgba(245,247,251,.88); }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3.5rem; }
        h1,h2,h3,h4 { color: var(--vision-ink); letter-spacing: -.035em; font-weight: 720; }
        h2 { margin-top: 1.2rem; margin-bottom: .35rem; }
        p,li,label,[data-testid="stCaptionContainer"] { line-height: 1.55; }
        [data-testid="stCaptionContainer"] { color: var(--vision-muted); }
        .vision-hero {
            position: relative; overflow: hidden; padding: 1.65rem 1.8rem;
            border-radius: 1.35rem; margin: .1rem 0 1.6rem; color: #f8fbff;
            background: linear-gradient(118deg,#14243b 0%,#1b3760 58%,#315efb 130%);
            box-shadow: 0 16px 36px rgba(21,36,58,.13);
        }
        .vision-hero::after {
            content: ""; position: absolute; width: 18rem; height: 18rem; right: -4.5rem; top: -9rem;
            border-radius: 50%; border: 1px solid rgba(255,255,255,.16);
            box-shadow: 0 0 0 2.2rem rgba(255,255,255,.035),0 0 0 4.4rem rgba(255,255,255,.025);
            pointer-events: none;
        }
        .vision-eyebrow { display: inline-flex; align-items: center; gap: .45rem; color: #c8d7ff; font-size: .72rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .65rem; }
        .vision-eyebrow-dot { display: inline-block; width: .45rem; height: .45rem; border-radius: 50%; background: #7ee0c3; box-shadow: 0 0 0 .22rem rgba(126,224,195,.15); }
        .vision-hero-title { margin: 0; color: #fff; font-size: clamp(2rem,4vw,2.8rem); font-weight: 780; letter-spacing: -.055em; line-height: 1.08; }
        .vision-hero-subtitle { margin-top: .7rem; max-width: 48rem; color: #d5e0f0; font-size: .98rem; line-height: 1.6; }
        .vision-hero-meta { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.1rem; }
        .vision-chip { display: inline-flex; align-items: center; padding: .34rem .65rem; border-radius: 999px; border: 1px solid rgba(255,255,255,.2); color: #edf3ff; background: rgba(255,255,255,.075); font-size: .76rem; font-weight: 600; }
        .vision-section-label { margin: 1.45rem 0 .7rem; color: var(--vision-ink); font-size: 1.12rem; font-weight: 750; letter-spacing: -.025em; }
        .vision-section-description { color: var(--vision-muted); margin-top: -.25rem; margin-bottom: .8rem; font-size: .9rem; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid var(--vision-line); border-radius: 1rem; padding: 1rem 1.05rem; box-shadow: 0 4px 16px rgba(21,36,58,.035); }
        [data-testid="stMetricLabel"] { color: var(--vision-muted); font-weight: 650; }
        [data-testid="stMetricValue"] { color: var(--vision-ink); letter-spacing: -.04em; }
        div[data-testid="stButton"] > button,div[data-testid="stDownloadButton"] > button,div[data-testid="stFormSubmitButton"] > button {
            min-height: 2.7rem; border-radius: .78rem; font-weight: 680; border: 1px solid #d7e0ec;
            transition: border-color 120ms ease,transform 120ms ease,box-shadow 120ms ease;
        }
        div[data-testid="stButton"] > button:hover,div[data-testid="stDownloadButton"] > button:hover,div[data-testid="stFormSubmitButton"] > button:hover {
            border-color: #8ba5ff; box-shadow: 0 5px 14px rgba(49,94,251,.10); transform: translateY(-1px);
        }
        div[data-testid="stButton"] > button[kind="primary"],div[data-testid="stFormSubmitButton"] > button[kind="primary"] { background: var(--vision-blue); border-color: var(--vision-blue); color: #fff; }
        [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stSelectbox"] div[data-baseweb="select"] > div,[data-testid="stNumberInput"] input { border-radius: .72rem; }
        [data-testid="stFileUploader"] section { border: 1px dashed #b8c7dc; border-radius: 1rem; background: rgba(255,255,255,.72); }
        [data-testid="stFileUploader"] section:hover { border-color: var(--vision-blue); background: #f8faff; }
        [data-testid="stExpander"] { border: 1px solid var(--vision-line); border-radius: .95rem; background: rgba(255,255,255,.75); overflow: hidden; }
        [data-testid="stExpander"] summary { font-weight: 680; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--vision-line); }
        [data-testid="stTabs"] button[role="tab"] { border-radius: .65rem .65rem 0 0; font-weight: 650; }
        [data-testid="stAlert"] { border-radius: .9rem; }
        .vision-image-frame { border: 1px solid var(--vision-line); background: #fff; border-radius: 1rem; overflow: hidden; box-shadow: 0 7px 24px rgba(21,36,58,.055); padding: .45rem; }
        .vision-image-caption { color: var(--vision-muted); font-size: .8rem; font-weight: 700; letter-spacing: .045em; text-transform: uppercase; margin: .1rem 0 .65rem .1rem; }
        [data-testid="stSidebar"] { background: #eef2f8; border-right: 1px solid #dce4ef; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
        [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { letter-spacing: -.025em; }
        .vision-sidebar-brand { padding: .95rem 1rem; border: 1px solid #dce4ef; border-radius: 1rem; background: rgba(255,255,255,.82); margin-bottom: 1.2rem; }
        .vision-sidebar-brand-title { color: var(--vision-ink); font-size: 1rem; font-weight: 800; letter-spacing: -.025em; }
        .vision-sidebar-brand-subtitle { color: var(--vision-muted); font-size: .78rem; margin-top: .2rem; line-height: 1.45; }
        .vision-sidebar-note { border-left: 3px solid #8ba5ff; padding: .55rem .7rem; color: #596b83; background: rgba(255,255,255,.65); border-radius: 0 .55rem .55rem 0; font-size: .76rem; line-height: 1.5; }
        hr { border-color: var(--vision-line); margin: 1.2rem 0; }
        @media (max-width: 768px) { .block-container { padding-top: 1.2rem; padding-bottom: 2rem; } .vision-hero { padding: 1.25rem; border-radius: 1rem; } .vision-hero-title { font-size: 2rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_header():
    st.markdown(
        """
        <section class="vision-hero">
            <div class="vision-eyebrow"><span class="vision-eyebrow-dot"></span> COMPUTER VISION WORKSPACE</div>
            <h1 class="vision-hero-title">AI Vision Lab</h1>
            <div class="vision-hero-subtitle">Face analysis, profile matching, and context-aware AI explanations in one focused workspace.</div>
            <div class="vision-hero-meta">
                <span class="vision-chip">Face landmarks</span>
                <span class="vision-chip">Identity profiles</span>
                <span class="vision-chip">AI explanation</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="vision-sidebar-brand">
                <div class="vision-sidebar-brand-title">AI Vision Lab</div>
                <div class="vision-sidebar-brand-subtitle">Analysis controls and display preferences</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Input")
        source = st.radio("Image source", ["Camera", "Upload image"], key="ui_image_source", horizontal=True)
        st.divider()
        st.subheader("Display")
        show_landmarks = st.checkbox("Show landmark visualization", value=True, key="ui_show_landmarks")
        show_coordinates = st.checkbox("Show landmark coordinates", value=False, key="ui_show_coordinates")
        show_measurements = st.checkbox("Show measurement lines on image", value=True, key="ui_show_measurements")
        st.divider()
        st.markdown(
            """
            <div class="vision-sidebar-note">Images are processed in memory. Profile embeddings and user-entered profile details may be saved to the configured database.</div>
            """,
            unsafe_allow_html=True,
        )
    return source, show_landmarks, show_coordinates, show_measurements

def show_image(image, caption):
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
    safe_caption = html.escape(str(caption))
    st.markdown(f'<div class="vision-image-caption">{safe_caption}</div>', unsafe_allow_html=True)
    st.html(
        f"""
        <div class="vision-image-frame">
          <img src="data:image/png;base64,{encoded}" alt="{safe_caption}" style="display:block;width:100%;height:auto;object-fit:contain;" />
        </div>
        """
    )
