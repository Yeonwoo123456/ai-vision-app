import base64
import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sqlalchemy import create_engine, text
from google import genai
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


def load_saved_analyses(engine):
    """Load all saved measurement rows so the CSV can accumulate across sessions."""
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT id, created_at, face_count, image_width, image_height,
                   face_index, landmark_count, eye_distance, face_width,
                   face_height, width_height_ratio, mouth_gap
            FROM analysis_history
            ORDER BY id ASC
        """))
        return pd.DataFrame(result.mappings().all())


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


def distance(p1, p2, image_width=1, image_height=1):
    """Return distance normalized by the image diagonal, accounting for aspect ratio."""
    dx = (p1["x"] - p2["x"]) * image_width
    dy = (p1["y"] - p2["y"]) * image_height
    diagonal = float(np.hypot(image_width, image_height))
    if diagonal <= 0:
        return None
    return float(np.hypot(dx, dy) / diagonal)


def center(landmarks, indices):
    points = [landmarks[i] for i in indices if i < len(landmarks)]

    if not points:
        return None

    return {
        "x": float(np.mean([p["x"] for p in points])),
        "y": float(np.mean([p["y"] for p in points])),
        "z": float(np.mean([p["z"] for p in points])),
    }


def analyze_face(face, image_width=1, image_height=1):
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
        distance(left_eye, right_eye, image_width, image_height)
        if left_eye and right_eye else None
    )
    face_width = distance(left, right, image_width, image_height) if left and right else None
    face_height = distance(top, bottom, image_width, image_height) if top and bottom else None
    mouth_gap = (
        distance(upper_lip, lower_lip, image_width, image_height)
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



def draw_measurement_overlay(image, face):
    """Draw color-coded facial regions and measurement guides on an RGB image."""
    image = validate_image(image).copy()
    height, width = image.shape[:2]
    points = face.get("landmarks", [])

    from PIL import ImageDraw

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)

    def xy(index):
        if index < 0 or index >= len(points):
            return None
        point = points[index]
        try:
            x = int(round(float(point["x"]) * width))
            y = int(round(float(point["y"]) * height))
        except (KeyError, TypeError, ValueError):
            return None
        return (max(0, min(width - 1, x)), max(0, min(height - 1, y)))

    def draw_chain(indices, color, line_width=None, closed=False):
        coords = [xy(i) for i in indices]
        coords = [pt for pt in coords if pt is not None]
        if len(coords) >= 2:
            if closed:
                coords.append(coords[0])
            draw.line(coords, fill=color, width=line_width or max(2, width // 500), joint="curve")

    def draw_points(indices, color, radius=None):
        radius = radius or max(2, width // 250)
        for index in indices:
            pt = xy(index)
            if pt is not None:
                x, y = pt
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=(255, 255, 255), width=1)

    # MediaPipe Face Mesh contours: distinct colors for the main facial regions.
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]
    left_brow = [70, 63, 105, 66, 107]
    right_brow = [336, 296, 334, 293, 300]
    outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                  375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    inner_lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                  324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    nose_bridge = [168, 6, 197, 195, 5, 4]
    nose_base = [98, 97, 2, 326, 327]

    draw_chain(face_oval, (255, 210, 40), closed=True)
    draw_chain(left_eye, (50, 150, 255), closed=True)
    draw_chain(right_eye, (50, 150, 255), closed=True)
    draw_chain(left_brow, (170, 100, 255))
    draw_chain(right_brow, (170, 100, 255))
    draw_chain(outer_lips, (255, 70, 100))
    draw_chain(inner_lips, (255, 70, 100))
    draw_chain(nose_bridge, (40, 220, 220))
    draw_chain(nose_base, (40, 220, 220))
    draw_points([33, 133, 362, 263], (50, 150, 255))
    draw_points([234, 454, 10, 152, 13, 14], (255, 210, 40))

    def group_center(indices):
        coords = [xy(i) for i in indices]
        coords = [pt for pt in coords if pt is not None]
        if not coords:
            return None
        return (int(sum(pt[0] for pt in coords) / len(coords)),
                int(sum(pt[1] for pt in coords) / len(coords)))

    measurements = [
        (group_center([33, 133, 159, 145]), group_center([362, 263, 386, 374]), "Eye distance", (40, 130, 255)),
        (xy(234), xy(454), "Face width", (30, 180, 80)),
        (xy(10), xy(152), "Face height", (255, 145, 0)),
        (xy(13), xy(14), "Mouth opening", (220, 50, 80)),
    ]

    line_width = max(2, width // 400)
    radius = max(3, width // 180)
    for start, end, label, color in measurements:
        if start is None or end is None:
            continue
        draw.line([start, end], fill=color, width=line_width)
        for x, y in (start, end):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white", width=1)

    return np.asarray(canvas)


def pct(value):
    return f"{value * 100:.2f}%" if value is not None else "N/A"


def get_gemini_settings():
    """Read Gemini credentials from Streamlit Secrets or environment variables."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        model = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
    except Exception:
        api_key = ""
        model = "gemini-3.8-flash"

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not model:
        model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    return api_key, model


def generate_ai_measurement_explanation(measurements, image_width, image_height):
    """Explain numeric measurements with Gemini; the image itself is never sent."""
    api_key, model = get_gemini_settings()
    if not api_key:
        raise ValueError("Gemini API key is not configured. Add GEMINI_API_KEY to Streamlit Secrets.")

    payload = {
        "measurements": {
            "eye_distance_relative": measurements.get("eye_distance"),
            "face_width_relative": measurements.get("face_width"),
            "face_height_relative": measurements.get("face_height"),
            "face_width_height_ratio": measurements.get("width_height_ratio"),
            "mouth_gap_relative": measurements.get("mouth_gap"),
            "landmark_count": measurements.get("landmark_count"),
            "face_index": measurements.get("face_index"),
        },
        "image_dimensions_pixels": {"width": image_width, "height": image_height},
        "measurement_note": "Relative distances are normalized by the image diagonal, not physical measurements.",
    }

    prompt = (
        "You are the AI Vision Lab measurement explainer. Explain the supplied computer-vision "
        "measurements clearly and conservatively. Only numeric measurements are provided; no image is sent. "
        "Explain what each metric represents and what can or cannot be inferred. Do not infer identity, age, "
        "ethnicity, attractiveness, personality, health, emotion, or other personal traits. Do not claim the "
        "values are centimeters or inches. Relative distances are normalized by the image diagonal and can "
        "be affected by pose, perspective, image quality, and landmark estimation error. Comparisons require "
        "consistent camera angle, pose, and image conditions. Use short headings and plain language. "
        "If a value is null, state that it was unavailable. Distinguish data from interpretation.\n\n"
        "Explain these measured results without inventing missing values:\n\n"
        + pd.Series(payload).to_json(force_ascii=False, indent=2)
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    output = (response.text or "").strip()
    if not output:
        raise ValueError("Gemini returned an empty explanation. Please try again.")
    return output


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

    show_measurements = st.checkbox(
        "Show measurement lines on image",
        value=True,
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

    analyses = [analyze_face(face, original.shape[1], original.shape[0]) for face in faces]


    if show_landmarks and show_measurements and faces:
        for detected_face in faces:
            annotated = draw_measurement_overlay(annotated, detected_face)

    if show_landmarks:
        st.subheader("Landmark Visualization")
        if show_measurements and faces:
            st.caption("Colored lines and endpoints show the measured facial landmarks; measurement labels are hidden.")
        show_image(annotated, "Landmark Visualization")

    if not faces:
        st.warning("No face was detected.")
    else:
        selected_index = st.selectbox(
            "Select face for AI explanation",
            range(len(faces)),
            format_func=lambda i: f"Face {i + 1}",
        )

        selected = analyses[selected_index]

        # Show only the AI's natural-language interpretation to the user.
        # Raw measurement metrics remain internal for AI explanation and database persistence.
        st.subheader("AI Measurement Explanation")
        st.caption(
            "The explanation is based on measured landmark values. The image itself is not sent to Gemini."
        )
        if st.button("Generate AI explanation", key="generate_ai_measurement_explanation"):
            try:
                with st.spinner("Generating explanation with Gemini..."):
                    explanation = generate_ai_measurement_explanation(
                        selected, original.shape[1], original.shape[0]
                    )
                st.session_state["ai_measurement_explanation"] = explanation
                st.session_state["ai_measurement_explanation_face"] = selected_index
            except Exception as exc:
                st.error(
                    "Could not generate the AI explanation. Check GEMINI_API_KEY in Streamlit Secrets, "
                    f"model access, and API limits. Details: {exc}"
                )

        if (
            st.session_state.get("ai_measurement_explanation")
            and st.session_state.get("ai_measurement_explanation_face") == selected_index
        ):
            st.markdown(st.session_state["ai_measurement_explanation"])

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

        # Keep database saving below the natural-language explanation.
        st.divider()
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


# Keep the accumulated database export independent of the current image analysis.
if db_engine is not None:
    st.subheader("Accumulated Analysis Data")
    try:
        saved_rows = load_saved_analyses(db_engine)
        if not saved_rows.empty:
            st.download_button(
                "Download accumulated saved data CSV",
                data=saved_rows.to_csv(index=False).encode("utf-8-sig"),
                file_name="analysis_history.csv",
                mime="text/csv",
                help="Downloads all analyses previously saved to the database.",
                key="download_accumulated_analysis_csv",
            )
        else:
            st.caption("No saved database records yet. Click 'Save analysis to database' after analyzing an image to accumulate records.")
    except Exception as exc:
        st.warning(f"Could not load accumulated saved data: {exc}")
