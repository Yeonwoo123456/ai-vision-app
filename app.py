import base64
import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sqlalchemy import create_engine, text
from detector import detect_faces


st.set_page_config(
    page_title="AI Vision Lab",
    layout="wide",
)



def get_database_url():
    """Read the Neon connection string from Streamlit Secrets or environment."""
    try:
        url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        url = ""

    if not url:
        url = os.environ.get("DATABASE_URL", "")

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


@st.cache_resource
def get_engine():
    url = get_database_url()
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)


def initialize_database(engine):
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id BIGSERIAL PRIMARY KEY,
                created_at TEXT NOT NULL,
                face_count INTEGER NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                face_index INTEGER NOT NULL,
                landmark_count INTEGER NOT NULL,
                eye_distance DOUBLE PRECISION,
                face_width DOUBLE PRECISION,
                face_height DOUBLE PRECISION,
                width_height_ratio DOUBLE PRECISION,
                mouth_gap DOUBLE PRECISION
            )
        """))


def save_analysis(engine, face_count, width, height, result):
    created_at = datetime.now(timezone.utc).isoformat()
    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO analysis_history (
                    created_at, face_count, image_width, image_height,
                    face_index, landmark_count, eye_distance, face_width,
                    face_height, width_height_ratio, mouth_gap
                ) VALUES (
                    :created_at, :face_count, :image_width, :image_height,
                    :face_index, :landmark_count, :eye_distance, :face_width,
                    :face_height, :width_height_ratio, :mouth_gap
                )
            """),
            {
                "created_at": created_at,
                "face_count": face_count,
                "image_width": width,
                "image_height": height,
                **result,
            },
        )

def load_image(uploaded_file):
    data = uploaded_file.getvalue()

    if not data:
        raise ValueError("The uploaded image is empty.")

    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        result = np.asarray(image, dtype=np.uint8)

    return np.ascontiguousarray(result)


def validate_image(image):
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Invalid image dimensions.")

    if image.dtype != np.uint8:
        image = np.clip(np.nan_to_num(image), 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def show_image(image, caption):
    image = validate_image(image)

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    st.markdown(f"**{caption}**")

    st.html(
        f"""
        <div style="width:100%;overflow:hidden;">
          <img
            src="data:image/png;base64,{encoded}"
            alt="Image preview"
            style="display:block;width:100%;height:auto;object-fit:contain;"
          />
        </div>
        """
    )


def distance(p1, p2):
    return float(np.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"]))


def center(landmarks, indices):
    points = [landmarks[i] for i in indices if i < len(landmarks)]

    if not points:
        return None

    return {
        "x": float(np.mean([p["x"] for p in points])),
        "y": float(np.mean([p["y"] for p in points])),
        "z": float(np.mean([p["z"] for p in points])),
    }


def analyze_face(face):
    points = face["landmarks"]

    def point(index):
        return points[index] if index < len(points) else None

    left_eye = center(points, [33, 133, 159, 145])
    right_eye = center(points, [362, 263, 386, 374])

    left = point(234)
    right = point(454)
    top = point(10)
    bottom = point(152)
    upper_lip = point(13)
    lower_lip = point(14)

    eye_distance = (
        distance(left_eye, right_eye)
        if left_eye and right_eye else None
    )
    face_width = distance(left, right) if left and right else None
    face_height = distance(top, bottom) if top and bottom else None
    mouth_gap = (
        distance(upper_lip, lower_lip)
        if upper_lip and lower_lip else None
    )

    ratio = (
        face_width / face_height
        if face_width is not None
        and face_height is not None
        and face_height > 0
        else None
    )

    return {
        "face_index": face["face_index"],
        "landmark_count": face["landmark_count"],
        "eye_distance": eye_distance,
        "face_width": face_width,
        "face_height": face_height,
        "width_height_ratio": ratio,
        "mouth_gap": mouth_gap,
    }


def pct(value):
    return f"{value * 100:.2f}%" if value is not None else "N/A"


st.title("AI Vision Lab")
st.caption("Facial Landmark Analysis")

with st.sidebar:
    st.header("Settings")

    source = st.radio(
        "Image source",
        ["Camera", "Upload image"],
    )

    show_landmarks = st.checkbox(
        "Show landmark visualization",
        value=True,
    )

    show_coordinates = st.checkbox(
        "Show landmark coordinates",
        value=False,
    )


uploaded_file = (
    st.camera_input("Take a picture")
    if source == "Camera"
    else st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )
)

# Connect to the database for saving only; no history table or graph is displayed.
db_engine = None
db_error = None
try:
    db_engine = get_engine()
    if db_engine is not None:
        initialize_database(db_engine)
except Exception as exc:
    db_error = str(exc)
    db_engine = None

if db_engine is None:
    st.caption("Database saving is unavailable until DATABASE_URL is configured correctly.")
    if db_error:
        st.warning(f"Database connection failed: {db_error}")

if uploaded_file is not None:
    try:
        original = load_image(uploaded_file)

        with st.spinner("Analyzing image..."):
            annotated, faces = detect_faces(original)

        original = validate_image(original)
        annotated = validate_image(annotated)

        if annotated.shape != original.shape:
            raise ValueError("Image dimensions changed during processing.")

    except Exception as exc:
        st.error(f"Image processing failed: {exc}")
        st.stop()

    analyses = [analyze_face(face) for face in faces]

    st.subheader("Analysis Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faces detected", len(faces))
    c2.metric(
        "Total landmarks",
        sum(face["landmark_count"] for face in faces),
    )
    c3.metric("Width", f"{original.shape[1]} px")
    c4.metric("Height", f"{original.shape[0]} px")

    left_col, right_col = st.columns(2)

    with left_col:
        show_image(original, "Original image")

    with right_col:
        if show_landmarks:
            show_image(annotated, "Landmark visualization")

    if not faces:
        st.warning("No face was detected.")
    else:
        st.subheader("Feature Measurements")

        selected_index = st.selectbox(
            "Select face",
            range(len(faces)),
            format_func=lambda i: f"Face {i + 1}",
        )

        selected = analyses[selected_index]

        a, b, c, d = st.columns(4)
        a.metric("Eye distance", pct(selected["eye_distance"]))
        b.metric("Face width", pct(selected["face_width"]))
        c.metric("Face height", pct(selected["face_height"]))
        d.metric(
            "Width / height",
            f'{selected["width_height_ratio"]:.3f}'
            if selected["width_height_ratio"] is not None
            else "N/A",
        )

        e, f = st.columns(2)
        e.metric("Mouth opening", pct(selected["mouth_gap"]))
        f.metric("Landmarks", selected["landmark_count"])

        rows = []

        for result in analyses:
            rows.append({
                "Face": result["face_index"] + 1,
                "Landmarks": result["landmark_count"],
                "Eye distance (%)": (
                    round(result["eye_distance"] * 100, 2)
                    if result["eye_distance"] is not None else None
                ),
                "Face width (%)": (
                    round(result["face_width"] * 100, 2)
                    if result["face_width"] is not None else None
                ),
                "Face height (%)": (
                    round(result["face_height"] * 100, 2)
                    if result["face_height"] is not None else None
                ),
                "Width/height ratio": result["width_height_ratio"],
                "Mouth opening (%)": (
                    round(result["mouth_gap"] * 100, 2)
                    if result["mouth_gap"] is not None else None
                ),
            })

        summary = pd.DataFrame(rows)

        st.subheader("Face Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Always display the Save button. If the database is unavailable,
        # show a clear message after the user clicks it instead of hiding it.
        if st.button("Save analysis to database", key="save_analysis_to_database"):
            if db_engine is None:
                st.error(
                    "Cannot save because the database is not connected. "
                    "Check DATABASE_URL in Streamlit Secrets and the connection error above."
                )
            else:
                try:
                    for result in analyses:
                        save_analysis(
                            db_engine,
                            len(faces),
                            original.shape[1],
                            original.shape[0],
                            result,
                        )
                    st.success("Analysis summary saved to database.")
                except Exception as exc:
                    st.error(f"Unable to save analysis: {exc}")

        st.download_button(
            "Download analysis CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="analysis_summary.csv",
            mime="text/csv",
        )

        if show_coordinates:
            coordinates = pd.DataFrame(
                faces[selected_index]["landmarks"]
            )
            coordinates.insert(
                0,
                "landmark_index",
                range(len(coordinates)),
            )

            st.subheader("Landmark Coordinates")
            st.dataframe(
                coordinates,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download coordinates CSV",
                data=coordinates.to_csv(index=False).encode("utf-8"),
                file_name="landmark_coordinates.csv",
                mime="text/csv",
            )
