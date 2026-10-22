import base64
import hashlib
import io
import json
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
                mouth_gap DOUBLE PRECISION,
                profile_id BIGINT NULL
            )
        """))
        # Safe migration for existing Neon databases created before profile linking.
        connection.execute(text("ALTER TABLE analysis_history ADD COLUMN IF NOT EXISTS profile_id BIGINT NULL"))
        # Identity profiles are separate from measurement history. Only embeddings and
        # user-entered profile text are stored; source images are not persisted.
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS face_profiles (
                id BIGSERIAL PRIMARY KEY,
                alias TEXT NOT NULL,
                name TEXT,
                description TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        # Backward-compatible migration: old profiles used alias; new UI uses name.
        connection.execute(text("ALTER TABLE face_profiles ADD COLUMN IF NOT EXISTS name TEXT"))
        connection.execute(text("UPDATE face_profiles SET name = alias WHERE name IS NULL OR name = ''"))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS face_profile_embeddings (
                id BIGSERIAL PRIMARY KEY,
                profile_id BIGINT NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS face_match_reviews (
                id BIGSERIAL PRIMARY KEY,
                profile_id BIGINT NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                image_fingerprint TEXT NOT NULL,
                decision TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))


def save_analysis(engine, face_count, width, height, result, profile_id=None):
    created_at = datetime.now(timezone.utc).isoformat()
    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO analysis_history (
                    created_at, face_count, image_width, image_height,
                    face_index, landmark_count, eye_distance, face_width,
                    face_height, width_height_ratio, mouth_gap, profile_id
                ) VALUES (
                    :created_at, :face_count, :image_width, :image_height,
                    :face_index, :landmark_count, :eye_distance, :face_width,
                    :face_height, :width_height_ratio, :mouth_gap, :profile_id
                )
            """),
            {
                "created_at": created_at,
                "face_count": face_count,
                "image_width": width,
                "image_height": height,
                **result,
                "profile_id": profile_id,
            },
        )


def load_saved_analyses(engine):
    """Load all saved measurement rows so the CSV can accumulate across sessions."""
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT h.id, h.created_at, h.face_count, h.image_width, h.image_height,
                   h.face_index, h.landmark_count, h.eye_distance, h.face_width,
                   h.face_height, h.width_height_ratio, h.mouth_gap, h.profile_id,
                   COALESCE(NULLIF(p.name, ''), p.alias) AS profile_name
            FROM analysis_history h
            LEFT JOIN face_profiles p ON p.id = h.profile_id
            ORDER BY h.id ASC
        """))
        return pd.DataFrame(result.mappings().all())


@st.cache_resource
def get_face_embedding_model():
    """Load InsightFace once. It detects faces and creates identity embeddings locally."""
    from insightface.app import FaceAnalysis

    model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return model


def get_identity_faces(image_rgb):
    """Return detected face embeddings from the current image; never saves the image."""
    image_rgb = validate_image(image_rgb)
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    detections = get_face_embedding_model().get(image_bgr)
    results = []
    for detected in detections:
        embedding = np.asarray(detected.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        results.append({
            "embedding": embedding,
            "bbox": [int(round(v)) for v in detected.bbox.tolist()],
            "det_score": float(getattr(detected, "det_score", 0.0)),
        })
    return results


def load_face_profiles(engine):
    with engine.connect() as connection:
        rows = connection.execute(text("""
            SELECT p.id, COALESCE(NULLIF(p.name, ''), p.alias) AS name, p.description, p.notes, e.embedding
            FROM face_profiles p
            LEFT JOIN face_profile_embeddings e ON e.profile_id = p.id
            ORDER BY p.id, e.id
        """)).mappings().all()
    profiles = {}
    for row in rows:
        profile_id = row["id"]
        if profile_id not in profiles:
            profiles[profile_id] = {
                "id": profile_id,
                "name": row["name"],
                "description": row["description"] or "",
                "notes": row["notes"] or "",
                "embeddings": [],
            }
        if row["embedding"]:
            try:
                profiles[profile_id]["embeddings"].append(
                    np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    return list(profiles.values())


def create_face_profile(engine, name, description, notes, embedding):
    now = datetime.now(timezone.utc).isoformat()
    normalized = np.asarray(embedding, dtype=np.float32)
    normalized = normalized / max(float(np.linalg.norm(normalized)), 1e-12)
    with engine.begin() as connection:
        profile_id = connection.execute(text("""
            INSERT INTO face_profiles (alias, name, description, notes, created_at, updated_at)
            VALUES (:alias, :name, :description, :notes, :created_at, :updated_at)
            RETURNING id
        """), {
            "alias": name.strip(), "name": name.strip(), "description": description.strip(), "notes": notes.strip(),
            "created_at": now, "updated_at": now,
        }).scalar_one()
        connection.execute(text("""
            INSERT INTO face_profile_embeddings (profile_id, embedding, created_at)
            VALUES (:profile_id, :embedding, :created_at)
        """), {
            "profile_id": profile_id,
            "embedding": json.dumps(normalized.tolist()),
            "created_at": now,
        })
    return profile_id


def add_face_profile_sample(engine, profile_id, embedding):
    now = datetime.now(timezone.utc).isoformat()
    normalized = np.asarray(embedding, dtype=np.float32)
    normalized = normalized / max(float(np.linalg.norm(normalized)), 1e-12)
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO face_profile_embeddings (profile_id, embedding, created_at)
            VALUES (:profile_id, :embedding, :created_at)
        """), {
            "profile_id": int(profile_id),
            "embedding": json.dumps(normalized.tolist()),
            "created_at": now,
        })
        connection.execute(text("""
            UPDATE face_profiles SET updated_at = :updated_at WHERE id = :profile_id
        """), {"updated_at": now, "profile_id": int(profile_id)})


def delete_face_profile(engine, profile_id):
    with engine.begin() as connection:
        # Preserve the accumulated measurement history; remove only its profile link.
        connection.execute(text("UPDATE analysis_history SET profile_id = NULL WHERE profile_id = :id"), {"id": int(profile_id)})
        connection.execute(text("DELETE FROM face_match_reviews WHERE profile_id = :id"), {"id": int(profile_id)})
        connection.execute(text("DELETE FROM face_profiles WHERE id = :id"), {"id": int(profile_id)})


def find_face_profile_matches(embedding, profiles):
    """Rank candidate profiles by cosine similarity; scores are suggestions, not proof."""
    query = np.asarray(embedding, dtype=np.float32)
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    matches = []
    for profile in profiles:
        scores = []
        for sample in profile["embeddings"]:
            sample = np.asarray(sample, dtype=np.float32)
            sample = sample / max(float(np.linalg.norm(sample)), 1e-12)
            if sample.shape == query.shape:
                scores.append(float(np.dot(query, sample)))
        if scores:
            matches.append({**profile, "similarity": max(scores)})
    return sorted(matches, key=lambda item: item["similarity"], reverse=True)


def get_image_fingerprint(uploaded_file):
    """Ephemeral identifier used to avoid attaching a confirmed profile to another upload."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def map_identity_face_to_measurement_face(identity_face, measurement_faces, image_width, image_height):
    """Map detector outputs by normalized face-box center, not by assumed list ordering."""
    if not measurement_faces:
        return None
    x1, y1, x2, y2 = identity_face["bbox"]
    target_x = ((x1 + x2) / 2.0) / max(image_width, 1)
    target_y = ((y1 + y2) / 2.0) / max(image_height, 1)
    best_index, best_distance = None, float("inf")
    for i, face in enumerate(measurement_faces):
        points = face.get("landmarks", [])
        if not points:
            continue
        xs = [float(pt.get("x", 0.0)) for pt in points]
        ys = [float(pt.get("y", 0.0)) for pt in points]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        distance = float(np.hypot(target_x - center_x, target_y - center_y))
        if distance < best_distance:
            best_index, best_distance = i, distance
    # Avoid linking records if the two detectors' face locations disagree substantially.
    return best_index if best_index is not None and best_distance <= 0.18 else None


def record_profile_review(engine, profile_id, image_fingerprint, decision="user_confirmed_same_person"):
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO face_match_reviews (profile_id, image_fingerprint, decision, created_at)
            VALUES (:profile_id, :image_fingerprint, :decision, :created_at)
        """), {
            "profile_id": int(profile_id),
            "image_fingerprint": image_fingerprint,
            "decision": decision,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })


def load_profile_analyses(engine, profile_id):
    with engine.connect() as connection:
        rows = connection.execute(text("""
            SELECT h.id, h.created_at, h.face_count, h.image_width, h.image_height,
                   h.face_index, h.landmark_count, h.eye_distance, h.face_width,
                   h.face_height, h.width_height_ratio, h.mouth_gap,
                   h.profile_id, COALESCE(NULLIF(p.name, ''), p.alias) AS profile_name
            FROM analysis_history h
            JOIN face_profiles p ON p.id = h.profile_id
            WHERE h.profile_id = :profile_id
            ORDER BY h.id ASC
        """), {"profile_id": int(profile_id)}).mappings().all()
    return pd.DataFrame(rows)


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
        model = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
    except Exception:
        api_key = ""
        model = "gemini-2.5-flash"

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not model:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return api_key, model


def generate_ai_measurement_explanation(measurements, image_width, image_height, profile_context=None):
    """Explain current measurements in the context of a saved user-provided profile; no image is sent."""
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

    profile_context = profile_context or {}
    profile_payload = {
        "saved_profile_name": profile_context.get("name"),
        "user_provided_description": profile_context.get("description", ""),
        "user_provided_notes": profile_context.get("notes", ""),
        "match_similarity_score": profile_context.get("similarity"),
        "profile_match_status": profile_context.get("status", "no confirmed profile match"),
    }
    prompt = (
        "You are the AI Vision Lab profile-aware explanation assistant. Explain the newly submitted face's "
        "available landmark measurements in the context of the saved profile information below. The image "
        "itself is NOT sent to you. The profile name, description, and notes are user-provided metadata, not "
        "verified facts. Clearly distinguish stored user-provided information from the current image's numeric "
        "measurements. If a candidate profile is unconfirmed, describe it only as a possible match and never "
        "state that it is definitely the same person. Similarity score is model-specific and is not a probability. "
        "Do not infer identity, age, ethnicity, attractiveness, personality, health, emotion, or other sensitive "
        "personal traits. Do not invent visual details because you cannot see the image. Do not claim values are "
        "centimeters or inches. Relative distances are normalized by the image diagonal and affected by pose, "
        "perspective, image quality, and landmark estimation error. Use concise plain language with headings: "
        "Saved profile context, Current measurement summary, and Limitations. If there is no saved profile match, "
        "say so. If a value is null, state it was unavailable.\n\n"
        "Saved profile context:\n"
        + json.dumps(profile_payload, ensure_ascii=False, indent=2)
        + "\n\nCurrent image measurements (no image provided):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
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
        current_image_fingerprint = get_image_fingerprint(uploaded_file)

        # A profile confirmation is valid only for the exact current upload.
        if st.session_state.get("confirmed_profile_image_fingerprint") != current_image_fingerprint:
            st.session_state.pop("confirmed_profile_id_for_current_image", None)
            st.session_state.pop("confirmed_profile_face_index", None)
            st.session_state.pop("confirmed_profile_identity_index", None)

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
                with st.spinner("Generating profile-aware explanation with Gemini..."):
                    profile_context = {}
                    if db_engine is not None:
                        identity_faces_for_ai = get_identity_faces(original)
                        saved_profiles_for_ai = load_face_profiles(db_engine)
                        matched_identity = None
                        for identity_face in identity_faces_for_ai:
                            mapped_index = map_identity_face_to_measurement_face(
                                identity_face, faces, original.shape[1], original.shape[0]
                            )
                            if mapped_index == selected_index:
                                matched_identity = identity_face
                                break
                        if matched_identity is not None:
                            confirmed_profile_id = st.session_state.get("confirmed_profile_id_for_current_image")
                            confirmed_profile_fingerprint = st.session_state.get("confirmed_profile_image_fingerprint")
                            confirmed_measurement_index = st.session_state.get("confirmed_profile_face_index")
                            confirmed_identity_index = st.session_state.get("confirmed_profile_identity_index")
                            active_identity_index = next(
                                (i for i, identity_face in enumerate(identity_faces_for_ai)
                                 if identity_face is matched_identity), None
                            )
                            confirmed_profile = next(
                                (profile for profile in saved_profiles_for_ai
                                 if profile["id"] == confirmed_profile_id), None
                            )
                            if (
                                confirmed_profile is not None
                                and confirmed_profile_fingerprint == current_image_fingerprint
                                and confirmed_measurement_index == selected_index
                                and confirmed_identity_index == active_identity_index
                            ):
                                profile_context = {
                                    "name": confirmed_profile["name"],
                                    "description": confirmed_profile["description"],
                                    "notes": confirmed_profile["notes"],
                                    "similarity": None,
                                    "status": "user-confirmed match for this exact uploaded image and selected face",
                                }
                            else:
                                profile_matches_for_ai = find_face_profile_matches(
                                    matched_identity["embedding"], saved_profiles_for_ai
                                )
                                if profile_matches_for_ai:
                                    top_profile = profile_matches_for_ai[0]
                                    if top_profile["similarity"] >= 0.45:
                                        profile_context = {
                                            "name": top_profile["name"],
                                            "description": top_profile["description"],
                                            "notes": top_profile["notes"],
                                            "similarity": top_profile["similarity"],
                                            "status": "possible match; not user-confirmed for this image",
                                        }
                    explanation = generate_ai_measurement_explanation(
                        selected, original.shape[1], original.shape[0], profile_context
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
                    confirmed_profile_id = st.session_state.get("confirmed_profile_id_for_current_image")
                    confirmed_face_index = st.session_state.get("confirmed_profile_face_index")
                    confirmed_identity_index = st.session_state.get("confirmed_profile_identity_index")
                    active_identity_index = st.session_state.get("identity_face_select")
                    # A confirmation only applies to the exact upload and selected face.
                    if confirmed_identity_index != active_identity_index:
                        confirmed_profile_id = None
                        confirmed_face_index = None
                    linked_rows = 0
                    for index, result in enumerate(analyses):
                        row_profile_id = (
                            confirmed_profile_id
                            if confirmed_profile_id is not None and index == confirmed_face_index
                            else None
                        )
                        save_analysis(
                            db_engine,
                            len(faces),
                            original.shape[1],
                            original.shape[0],
                            result,
                            profile_id=row_profile_id,
                        )
                        if row_profile_id is not None:
                            linked_rows += 1
                    if linked_rows:
                        st.success(f"Analysis saved. {linked_rows} face record linked to the confirmed profile.")
                    else:
                        st.success("Analysis saved. No profile was linked because no same-person match was confirmed for this image.")
                except Exception as exc:
                    st.error(f"Unable to save analysis: {exc}")


# Same-person profile matching is an opt-in, separate feature. The raw image is processed
# locally; only the numeric embedding and user-entered name/description/notes are persisted.
if uploaded_file is not None and db_engine is not None:
    st.divider()
    st.subheader("Face Identity Profiles")
    st.caption(
        "Use only photos you have permission to process. Matches are suggestions, not proof of identity. "
        "The source image is not saved; profile embeddings and your notes are stored in Neon."
    )
    try:
        with st.spinner("Creating local face embeddings..."):
            identity_faces = get_identity_faces(original)
        profiles = load_face_profiles(db_engine)

        if not identity_faces:
            st.info("The identity model did not detect a face in this image.")
        else:
            identity_index = st.selectbox(
                "Select face for profile matching",
                range(len(identity_faces)),
                format_func=lambda i: f"Detected face {i + 1}",
                key="identity_face_select",
            )
            selected_identity = identity_faces[identity_index]
            matched_measurement_index = map_identity_face_to_measurement_face(
                selected_identity, faces, original.shape[1], original.shape[0]
            )
            matches = find_face_profile_matches(selected_identity["embedding"], profiles)
            # Conservative initial threshold; should be calibrated with consented test images.
            candidate = matches[0] if matches and matches[0]["similarity"] >= 0.45 else None

            if candidate:
                st.markdown("**Possible saved profile**")
                st.write(f"Name: {candidate['name']}")
                if candidate["description"]:
                    st.write(f"Description: {candidate['description']}")
                if candidate["notes"]:
                    st.write(f"Notes: {candidate['notes']}")
                st.caption(
                    f"Similarity score: {candidate['similarity']:.3f}. "
                    "This score is model-specific and must not be treated as a probability. Confirm the match before linking this sample."
                )
            elif matches:
                st.info("No strong candidate reached the initial matching threshold. You can review the closest candidates below.")
                for match in matches[:3]:
                    st.write(f"{match['name']} — similarity {match['similarity']:.3f}")
            else:
                st.info("No saved face profiles yet. Create a profile below to start your private gallery.")

            with st.expander("Create a new face profile", expanded=not bool(candidate)):
                with st.form("create_face_profile_form"):
                    new_name = st.text_input("Name (required)", max_chars=100)
                    new_description = st.text_area("Description", max_chars=1000)
                    new_notes = st.text_area("Notes", max_chars=3000)
                    confirm_save = st.checkbox("I have permission to save and compare this face sample.")
                    create_submitted = st.form_submit_button("Save new face profile")
                if create_submitted:
                    if not new_name.strip():
                        st.error("Enter a name before saving.")
                    elif not confirm_save:
                        st.error("Confirm that you have permission to process this face sample.")
                    else:
                        try:
                            profile_id = create_face_profile(
                                db_engine, new_name, new_description, new_notes,
                                selected_identity["embedding"],
                            )
                            record_profile_review(db_engine, profile_id, current_image_fingerprint, "user_created_profile")
                            st.session_state["confirmed_profile_id_for_current_image"] = profile_id
                            st.session_state["confirmed_profile_image_fingerprint"] = current_image_fingerprint
                            st.session_state["confirmed_profile_face_index"] = matched_measurement_index
                            st.session_state["confirmed_profile_identity_index"] = identity_index
                            st.success(f"Profile '{new_name.strip()}' saved (ID {profile_id}).")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not save face profile: {exc}")

            if profiles:
                st.markdown("**Manage existing profiles**")
                profile_by_id = {profile["id"]: profile for profile in profiles}
                selected_profile_id = st.selectbox(
                    "Choose profile to add this face sample to",
                    list(profile_by_id.keys()),
                    format_func=lambda profile_id: profile_by_id[profile_id]["name"],
                    key="profile_sample_target",
                )
                confirm_link = st.checkbox(
                    "I reviewed the candidate and confirm this sample belongs to the selected profile.",
                    key="confirm_profile_link",
                )
                if st.button("Add face sample to selected profile", key="add_face_profile_sample"):
                    if not confirm_link:
                        st.error("Confirm the match before adding a sample.")
                    else:
                        try:
                            add_face_profile_sample(
                                db_engine, selected_profile_id, selected_identity["embedding"]
                            )
                            record_profile_review(db_engine, selected_profile_id, current_image_fingerprint)
                            st.session_state["confirmed_profile_id_for_current_image"] = int(selected_profile_id)
                            st.session_state["confirmed_profile_image_fingerprint"] = current_image_fingerprint
                            st.session_state["confirmed_profile_face_index"] = matched_measurement_index
                            st.session_state["confirmed_profile_identity_index"] = identity_index
                            if matched_measurement_index is None:
                                st.warning("The identity detector and landmark detector could not confidently map this face to a measurement row. The embedding was saved, but this image's measurement row will not be linked to the profile.")
                            else:
                                st.success("Confirmed sample added. Save analysis to database to link this image's measurement record to the profile.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not add face sample: {exc}")

                st.markdown("**Export this profile's accumulated analysis CSV**")
                profile_rows = load_profile_analyses(db_engine, selected_profile_id)
                if not profile_rows.empty:
                    st.download_button(
                        "Download selected profile CSV",
                        data=profile_rows.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"face_profile_{selected_profile_id}_analysis.csv",
                        mime="text/csv",
                        key="download_profile_csv",
                    )
                else:
                    st.caption("No analysis records have been linked to this profile yet. Confirm a face sample, then click 'Save analysis to database'.")

                st.markdown("**Delete a profile**")
                delete_profile_id = st.selectbox(
                    "Choose profile to delete",
                    list(profile_by_id.keys()),
                    format_func=lambda profile_id: profile_by_id[profile_id]["name"],
                    key="delete_profile_target",
                )
                confirm_delete = st.checkbox(
                    "I understand this permanently deletes the selected profile and its saved embeddings.",
                    key="confirm_delete_profile",
                )
                if st.button("Delete selected face profile", key="delete_face_profile"):
                    if not confirm_delete:
                        st.error("Confirm deletion first.")
                    else:
                        try:
                            delete_face_profile(db_engine, delete_profile_id)
                            st.success("Face profile deleted.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not delete face profile: {exc}")
    except Exception as exc:
        st.warning(
            "Face profile matching is unavailable. Confirm that insightface and onnxruntime "
            f"are installed and the model can load. Details: {exc}"
        )


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
                help="Downloads all saved analysis records, including confirmed profile ID and profile name when linked.",
                key="download_accumulated_analysis_csv",
            )
        else:
            st.caption("No saved database records yet. Click 'Save analysis to database' after analyzing an image to accumulate records.")
    except Exception as exc:
        st.warning(f"Could not load accumulated saved data: {exc}")
