import hashlib
import heapq
import io
import itertools
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sqlalchemy import create_engine, text
from detector import detect_faces
from ui import configure_page, render_header, render_sidebar, show_image

configure_page()

def get_database_url():
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

        connection.execute(text("ALTER TABLE analysis_history ADD COLUMN IF NOT EXISTS profile_id BIGINT NULL"))

        connection.execute(text("ALTER TABLE analysis_history ADD COLUMN IF NOT EXISTS image_fingerprint TEXT NULL"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_analysis_image_face ON analysis_history (image_fingerprint, face_index) WHERE image_fingerprint IS NOT NULL"))

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

        connection.execute(text("ALTER TABLE face_profiles ADD COLUMN IF NOT EXISTS name TEXT"))
        connection.execute(text("UPDATE face_profiles SET name = alias WHERE name IS NULL OR name = ''"))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS face_profile_embeddings (
                id BIGSERIAL PRIMARY KEY,
                profile_id BIGINT NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sample_fingerprint TEXT NULL,
                sample_face_index INTEGER NULL
            )
        """))
        connection.execute(text("ALTER TABLE face_profile_embeddings ADD COLUMN IF NOT EXISTS sample_fingerprint TEXT NULL"))
        connection.execute(text("ALTER TABLE face_profile_embeddings ADD COLUMN IF NOT EXISTS sample_face_index INTEGER NULL"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_profile_sample_fingerprint ON face_profile_embeddings (profile_id, sample_fingerprint, sample_face_index) WHERE sample_fingerprint IS NOT NULL AND sample_face_index IS NOT NULL"))

def save_analysis(engine, face_count, width, height, result, profile_id=None, image_fingerprint=None):
    created_at = datetime.now(timezone.utc).isoformat()
    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO analysis_history (
                    created_at, face_count, image_width, image_height,
                    face_index, landmark_count, eye_distance, face_width,
                    face_height, width_height_ratio, mouth_gap, profile_id, image_fingerprint
                ) VALUES (
                    :created_at, :face_count, :image_width, :image_height,
                    :face_index, :landmark_count, :eye_distance, :face_width,
                    :face_height, :width_height_ratio, :mouth_gap, :profile_id, :image_fingerprint
                )
                ON CONFLICT (image_fingerprint, face_index) WHERE image_fingerprint IS NOT NULL
                DO UPDATE SET
                    profile_id = COALESCE(EXCLUDED.profile_id, analysis_history.profile_id)
            """),
            {
                "created_at": created_at,
                "face_count": int(face_count),
                "image_width": int(width),
                "image_height": int(height),
                **result,
                "profile_id": int(profile_id) if profile_id is not None else None,
                "image_fingerprint": image_fingerprint,
            },
        )

def load_saved_analyses(engine):
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
    from insightface.app import FaceAnalysis

    model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return model

def assess_face_quality(image_rgb, bbox, det_score, keypoints=None):
    import cv2

    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    face_width = max(0, x2 - x1)
    face_height = max(0, y2 - y1)
    crop = image_rgb[y1:y2, x1:x2]
    blur_score = 0.0
    if crop.size and crop.shape[0] >= 2 and crop.shape[1] >= 2:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []
    gray_crop = None
    brightness_mean = None
    brightness_std = None
    if crop.size and crop.shape[0] >= 2 and crop.shape[1] >= 2:
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        brightness_mean = float(np.mean(gray_crop))
        brightness_std = float(np.std(gray_crop))

    face_area_ratio = float((face_width * face_height) / max(width * height, 1))
    if min(face_width, face_height) < 90 or face_area_ratio < 0.012:
        warnings.append("Face is small in the image; use a closer, higher-resolution photo.")
    if float(det_score) < 0.60:
        warnings.append("Face detection confidence is low; try a clearer photo.")
    if blur_score < 25.0:
        warnings.append("The face may be blurry; use a sharper image.")
    if brightness_mean is not None and brightness_mean < 45:
        warnings.append("The face crop is very dark; use more even lighting.")
    elif brightness_mean is not None and brightness_mean > 215:
        warnings.append("The face crop may be overexposed; reduce strong lighting or glare.")
    if brightness_std is not None and brightness_std < 18:
        warnings.append("The face crop has very little contrast; use a clearer photo with balanced lighting.")

    pose_offset = None
    eye_roll_degrees = None
    if keypoints is not None:
        try:
            pts = np.asarray(keypoints, dtype=np.float32)
            if pts.shape[0] >= 3:
                eye_delta = pts[1] - pts[0]
                eye_distance = float(np.linalg.norm(eye_delta))
                eye_mid = (pts[0] + pts[1]) / 2.0
                if eye_distance > 1e-6:
                    pose_offset = float(abs(pts[2][0] - eye_mid[0]) / eye_distance)
                    raw_eye_angle = abs(float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))))

                    eye_roll_degrees = min(raw_eye_angle, abs(180.0 - raw_eye_angle))
                    if pose_offset > 0.22:
                        warnings.append("The face may be turned to the side; a more front-facing photo may match better.")
                    if eye_roll_degrees > 15:
                        warnings.append("The head appears tilted; a more level photo may improve matching.")
        except (TypeError, ValueError, IndexError):
            pose_offset = None
            eye_roll_degrees = None

    quality_ok = (
        min(face_width, face_height) >= 80
        and face_area_ratio >= 0.008
        and float(det_score) >= 0.45
        and blur_score >= 12.0
        and (brightness_mean is None or 25 <= brightness_mean <= 235)
        and (pose_offset is None or pose_offset <= 0.40)
        and (eye_roll_degrees is None or eye_roll_degrees <= 30)
    )
    return {
        "face_width_px": face_width,
        "face_height_px": face_height,
        "face_area_ratio": face_area_ratio,
        "det_score": float(det_score),
        "blur_score": blur_score,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "pose_offset": pose_offset,
        "eye_roll_degrees": eye_roll_degrees,
        "quality_ok": quality_ok,
        "warnings": warnings,
    }

def get_identity_faces(image_rgb):
    image_rgb = validate_image(image_rgb)
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    detections = get_face_embedding_model().get(image_bgr)
    results = []
    for detected in detections:
        embedding = np.asarray(detected.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        bbox = [int(round(v)) for v in detected.bbox.tolist()]
        det_score = float(getattr(detected, "det_score", 0.0))
        keypoints = getattr(detected, "kps", None)
        quality = assess_face_quality(image_rgb, bbox, det_score, keypoints)
        results.append({
            "embedding": embedding,
            "bbox": bbox,
            "det_score": det_score,
            "quality": quality,
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
                vector = normalize_embedding(json.loads(row["embedding"]))
                profiles[profile_id]["embeddings"].append(vector)
            except (ValueError, TypeError, json.JSONDecodeError):

                continue
    return list(profiles.values())

def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("The face model returned an invalid embedding. Please retry with another image.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("The face model returned an empty embedding. Please retry with another image.")
    return vector / norm

def create_face_profile(engine, name, description, notes, embedding, sample_fingerprint=None, sample_face_index=None):
    now = datetime.now(timezone.utc).isoformat()
    normalized = normalize_embedding(embedding)
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
            INSERT INTO face_profile_embeddings (
                profile_id, embedding, created_at, sample_fingerprint, sample_face_index
            ) VALUES (
                :profile_id, :embedding, :created_at, :sample_fingerprint, :sample_face_index
            )
            ON CONFLICT (profile_id, sample_fingerprint, sample_face_index)
            WHERE sample_fingerprint IS NOT NULL AND sample_face_index IS NOT NULL
            DO NOTHING
        """), {
            "profile_id": profile_id,
            "embedding": json.dumps(normalized.tolist()),
            "created_at": now,
            "sample_fingerprint": sample_fingerprint,
            "sample_face_index": sample_face_index,
        })
    return profile_id

def add_face_profile_sample(engine, profile_id, embedding, sample_fingerprint=None, sample_face_index=None):
    now = datetime.now(timezone.utc).isoformat()
    normalized = normalize_embedding(embedding)
    with engine.begin() as connection:
        result = connection.execute(text("""
            INSERT INTO face_profile_embeddings (
                profile_id, embedding, created_at, sample_fingerprint, sample_face_index
            ) VALUES (
                :profile_id, :embedding, :created_at, :sample_fingerprint, :sample_face_index
            )
            ON CONFLICT (profile_id, sample_fingerprint, sample_face_index)
            WHERE sample_fingerprint IS NOT NULL AND sample_face_index IS NOT NULL
            DO NOTHING
        """), {
            "profile_id": int(profile_id),
            "embedding": json.dumps(normalized.tolist()),
            "created_at": now,
            "sample_fingerprint": sample_fingerprint,
            "sample_face_index": sample_face_index,
        })
        if result.rowcount:
            connection.execute(text("""
                UPDATE face_profiles SET updated_at = :updated_at WHERE id = :profile_id
            """), {"updated_at": now, "profile_id": int(profile_id)})
        return bool(result.rowcount)

def update_face_profile_info(engine, profile_id, new_name="", description_addition="", notes_addition=""):
    now = datetime.now(timezone.utc).isoformat()
    with engine.begin() as connection:
        row = connection.execute(text("""
            SELECT name, alias, description, notes
            FROM face_profiles WHERE id = :profile_id
            FOR UPDATE
        """), {"profile_id": int(profile_id)}).mappings().first()
        if row is None:
            raise ValueError("The selected face profile no longer exists.")

        current_name = (row["name"] or row["alias"] or "").strip()
        current_description = (row["description"] or "").strip()
        current_notes = (row["notes"] or "").strip()
        proposed_name = (new_name or "").strip()
        description_addition = (description_addition or "").strip()
        notes_addition = (notes_addition or "").strip()

        if description_addition and description_addition not in current_description:
            current_description = "\n".join(part for part in [current_description, description_addition] if part)
        if notes_addition and notes_addition not in current_notes:
            current_notes = "\n".join(part for part in [current_notes, notes_addition] if part)

        final_name = proposed_name or current_name
        connection.execute(text("""
            UPDATE face_profiles
            SET name = :name, alias = :alias, description = :description,
                notes = :notes, updated_at = :updated_at
            WHERE id = :profile_id
        """), {
            "name": final_name,
            "alias": final_name,
            "description": current_description,
            "notes": current_notes,
            "updated_at": now,
            "profile_id": int(profile_id),
        })
    return final_name

def delete_face_profile(engine, profile_id):
    with engine.begin() as connection:

        connection.execute(text("UPDATE analysis_history SET profile_id = NULL WHERE profile_id = :id"), {"id": int(profile_id)})
        connection.execute(text("DELETE FROM face_profiles WHERE id = :id"), {"id": int(profile_id)})

def find_face_profile_matches(embedding, profiles):
    query = normalize_embedding(embedding)
    matches = []
    for profile in profiles:
        valid_samples = []
        for sample in profile["embeddings"]:
            try:
                normalized_sample = normalize_embedding(sample)
            except ValueError:
                continue
            if normalized_sample.shape == query.shape:
                valid_samples.append(normalized_sample)

        if not valid_samples:
            continue

        sample_matrix = np.vstack(valid_samples)
        scores = np.sort(sample_matrix @ query)[::-1]
        top_scores = scores[:min(3, len(scores))]
        best_score = float(top_scores[0])
        top_mean = float(np.mean(top_scores))
        aggregate_score = 0.65 * best_score + 0.35 * top_mean
        matches.append({
            **profile,
            "similarity": float(aggregate_score),
            "best_sample_similarity": best_score,
            "samples_compared": int(scores.size),
            "top_sample_mean": top_mean,
        })
    return sorted(matches, key=lambda item: item["similarity"], reverse=True)

def get_image_fingerprint(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()

def map_identity_face_to_measurement_face(identity_face, measurement_faces, image_width, image_height):
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

    return best_index if best_index is not None and best_distance <= 0.18 else None

def get_match_threshold():
    raw_value = None
    try:
        raw_value = st.secrets.get("MATCH_THRESHOLD", None)
    except Exception:
        raw_value = None
    if raw_value is None or raw_value == "":
        raw_value = os.environ.get("MATCH_THRESHOLD", "0.45")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 0.45
    return min(0.99, max(0.01, value))

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

def distance(p1, p2, image_width=1, image_height=1):
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

def draw_astar_path(image, landmarks, path):
    from PIL import ImageDraw

    canvas = Image.fromarray(validate_image(image).copy())
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    coords = []
    for index in path:
        point = landmarks[index]
        x = int(round(float(point["x"]) * width))
        y = int(round(float(point["y"]) * height))
        coords.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    if len(coords) >= 2:
        draw.line(coords, fill=(255, 80, 40), width=max(2, width // 400), joint="curve")
    radius = max(4, width // 180)
    for position, point in enumerate(coords):
        if position == 0:
            fill = (40, 220, 100)
        elif position == len(coords) - 1:
            fill = (255, 210, 40)
        else:
            fill = (255, 80, 40)
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)
    return np.asarray(canvas)

def generate_ai_measurement_explanation(profile_context=None):
    profile_context = profile_context or {}
    if profile_context.get("no_match") or profile_context.get("status") == "no saved profile match":
        return "No profile matching this person was found. If you would like, please create a new profile."

    name = str(profile_context.get("name") or "Not available").strip()
    description = str(profile_context.get("description") or "Not available").strip()
    note = str(profile_context.get("notes") or "Not available").strip()
    similarity = profile_context.get("similarity")
    status = str(profile_context.get("status", "no confirmed profile match"))

    if similarity is None:
        if "user-confirmed" in status.lower() or "user confirmed" in status.lower():
            similarity_text = "User-confirmed; score not available"
        elif "possible" in status.lower():
            similarity_text = "Not available (possible match; confirm before linking)"
        else:
            similarity_text = "Not available"
    else:
        try:
            score_text = f"{float(similarity):.3f}"
        except (TypeError, ValueError):
            score_text = str(similarity).strip()
        if "possible" in status.lower():
            similarity_text = f"{score_text} (model-specific score, not a probability; confirm before linking)"
        else:
            similarity_text = f"{score_text} (user-confirmed)"

    fields = {
        "Name": name,
        "Description": description,
        "Note": note,
        "Similarity": similarity_text,
    }
    return "\n\n".join(f"**{field}:** {value}" for field, value in fields.items())


def _landmark_vector(point):
    values = [float(point.get(axis, 0.0)) for axis in ("x", "y", "z")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Landmark coordinates must be finite numbers.")
    return np.asarray(values, dtype=np.float64)

def astar_landmark_path(landmarks, start, goal, k_neighbors=6):
    count = len(landmarks)
    if count < 2:
        raise ValueError("At least two landmarks are required.")
    if not 0 <= int(start) < count or not 0 <= int(goal) < count:
        raise IndexError("Start and goal must refer to existing landmarks.")
    if int(start) == int(goal):
        return {"path": [int(start)], "cost": 0.0, "expanded_nodes": 0}

    vectors = np.vstack([_landmark_vector(point) for point in landmarks])
    differences = vectors[:, None, :] - vectors[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    neighbors = max(1, min(int(k_neighbors), count - 1))

    graph = [set() for _ in range(count)]
    for node in range(count):
        nearest = np.argpartition(distances[node], neighbors)[:neighbors + 1]
        nearest = [int(other) for other in nearest if int(other) != node]
        nearest.sort(key=lambda other: distances[node, other])
        for other in nearest[:neighbors]:
            graph[node].add(other)
            graph[other].add(node)

    start, goal = int(start), int(goal)
    def heuristic(node):
        return float(np.linalg.norm(vectors[node] - vectors[goal]))

    counter = itertools.count()
    frontier = [(heuristic(start), next(counter), start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    expanded_nodes = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return {
                "path": path,
                "cost": float(g_score[goal]),
                "expanded_nodes": expanded_nodes,
            }

        closed.add(current)
        expanded_nodes += 1
        for neighbor in graph[current]:
            tentative_g = g_score[current] + float(distances[current, neighbor])
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                if neighbor in closed:
                    closed.remove(neighbor)
                heapq.heappush(
                    frontier,
                    (tentative_g + heuristic(neighbor), next(counter), neighbor),
                )

    raise ValueError("No path was found in the generated landmark graph.")

render_header()
source, show_landmarks, show_coordinates, show_measurements = render_sidebar()

uploaded_file = (
    st.camera_input("Take a picture")
    if source == "Camera"
    else st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )
)

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

        st.warning("Database connection failed. Check DATABASE_URL in Streamlit Secrets and review the private app logs for details.")

if uploaded_file is not None:
    try:
        original = load_image(uploaded_file)
        current_image_fingerprint = get_image_fingerprint(uploaded_file)

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

        with st.expander("A* Search: Landmark Graph Path Explorer", expanded=False):
            st.caption(
                "A* searches a k-nearest-neighbor graph built from this face's landmarks. "
                "Edge costs and the Euclidean heuristic use normalized 3D landmark coordinates. "
                "This exploratory route does not change identity matching or saved data."
            )
            landmark_points = faces[selected_index].get("landmarks", [])
            landmark_indices = list(range(len(landmark_points)))
            if len(landmark_indices) >= 2:
                default_start = 33 if 33 in landmark_indices else 0
                default_end = 263 if 263 in landmark_indices else len(landmark_indices) - 1
                path_col1, path_col2 = st.columns(2)
                with path_col1:
                    path_start = st.selectbox(
                        "Start landmark", landmark_indices,
                        index=default_start,
                        key=f"astar_start_{selected_index}_{current_image_fingerprint[:8]}",
                    )
                with path_col2:
                    path_goal = st.selectbox(
                        "Goal landmark", landmark_indices,
                        index=default_end,
                        key=f"astar_goal_{selected_index}_{current_image_fingerprint[:8]}",
                    )
                path_neighbors = st.slider(
                    "Graph neighbors per landmark", min_value=3, max_value=12, value=6,
                    help="Higher values create more connections and may shorten paths, but increase graph density.",
                    key=f"astar_neighbors_{selected_index}_{current_image_fingerprint[:8]}",
                )
                if st.button("Run A* Search", key=f"run_astar_{selected_index}_{current_image_fingerprint[:8]}"):
                    if path_start == path_goal:
                        st.info("Choose two different landmark indices to search for a path.")
                    else:
                        try:
                            path_result = astar_landmark_path(
                                landmark_points, path_start, path_goal, k_neighbors=path_neighbors
                            )
                            path_overlay = draw_astar_path(original, landmark_points, path_result["path"])
                            st.success(
                                f"A* found a path with {len(path_result['path'])} landmarks, "
                                f"cost {path_result['cost']:.5f}, and "
                                f"{path_result['expanded_nodes']} expanded nodes."
                            )
                            show_image(path_overlay, "A* landmark path")
                            st.caption(
                                "This is the shortest path on the generated k-nearest-neighbor graph, "
                                "not a medical or identity measurement."
                            )
                        except (ValueError, IndexError, KeyError) as exc:
                            st.warning(f"A* path search could not complete: {exc}")
            else:
                st.info("At least two detected landmarks are required for A* path search.")

        st.subheader("AI Measurement Explanation")
        st.caption(
            "Profile information is formatted locally from saved records; no Gemini request is made."
        )
        if st.button("Generate AI explanation", key="generate_ai_measurement_explanation"):
            try:
                with st.spinner("Formatting saved profile information..."):
                    profile_context = {"no_match": True, "status": "no saved profile match"}
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
                        if matched_identity is not None and matched_identity.get("quality", {}).get("quality_ok", False):
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
                                    if top_profile["similarity"] >= get_match_threshold():
                                        profile_context = {
                                            "name": top_profile["name"],
                                            "description": top_profile["description"],
                                            "notes": top_profile["notes"],
                                            "similarity": top_profile["similarity"],
                                            "status": "possible match; not user-confirmed for this image",
                                        }
                                    else:
                                        profile_context = {"no_match": True, "status": "no saved profile match"}
                                else:
                                    profile_context = {"no_match": True, "status": "no saved profile match"}
                        elif matched_identity is not None:
                            profile_context = {"no_match": True, "status": "low quality image; request clearer photo"}
                    explanation = generate_ai_measurement_explanation(profile_context)
                st.session_state["ai_measurement_explanation"] = explanation
                st.session_state["ai_measurement_explanation_face"] = selected_index
            except Exception as exc:
                st.error(f"Could not format the saved profile information: {exc}")

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
                            image_fingerprint=current_image_fingerprint,
                        )
                        if row_profile_id is not None:
                            linked_rows += 1
                    if linked_rows:
                        st.success(f"Analysis saved. {linked_rows} face record linked to the confirmed profile.")
                    else:
                        st.success("Analysis saved. No profile was linked because no same-person match was confirmed for this image.")
                except Exception:
                    st.error("Unable to save analysis. Check DATABASE_URL and the private app logs; connection details are not displayed here.")

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
            quality = selected_identity.get("quality", {})
            if quality.get("warnings"):
                st.warning("Face photo quality check: " + " ".join(quality["warnings"]))
            else:
                st.success("Face photo quality checks passed. Matching can still be uncertain.")
            with st.expander("Photo quality details", expanded=False):
                q1, q2, q3 = st.columns(3)
                q1.metric("Face crop", f'{quality.get("face_width_px", 0)} × {quality.get("face_height_px", 0)} px')
                q2.metric("Detection score", f'{quality.get("det_score", 0.0):.3f}')
                q3.metric("Blur score", f'{quality.get("blur_score", 0.0):.1f}')
                q4, q5, q6 = st.columns(3)
                brightness = quality.get("brightness_mean")
                q4.metric("Mean brightness", f"{brightness:.1f}/255" if brightness is not None else "N/A")
                q5.metric("Face image area", f'{quality.get("face_area_ratio", 0.0) * 100:.1f}%')
                roll = quality.get("eye_roll_degrees")
                q6.metric("Estimated head tilt", f"{roll:.1f}°" if roll is not None else "N/A")
                st.caption("These are heuristic image-quality indicators, not identity-confidence scores or precise pose measurements.")

            matches = find_face_profile_matches(selected_identity["embedding"], profiles) if quality.get("quality_ok") else []

            match_threshold = get_match_threshold()
            candidate = matches[0] if matches and matches[0]["similarity"] >= match_threshold else None
            if not quality.get("quality_ok"):
                st.info("Profile matching is paused for this image because quality is below the minimum screening threshold. Upload a clearer, larger, sharper, more front-facing photo.")
            elif profiles and not candidate:
                st.caption(f"No profile reached the current candidate threshold ({match_threshold:.2f}). This does not prove the person is new.")

            if profiles:
                st.markdown("**Manage existing profiles**")
                profile_by_id = {profile["id"]: profile for profile in profiles}
                selected_profile_id = st.selectbox(
                    "Choose existing profile",
                    list(profile_by_id.keys()),
                    format_func=lambda profile_id: profile_by_id[profile_id]["name"],
                    key="profile_sample_target",
                )
                selected_profile = profile_by_id[selected_profile_id]

                with st.expander("Update / Delete existing profile", expanded=True):
                    st.caption(
                        "After confirming this is the same person, the new face sample is added and new information is appended. "
                        "Existing notes are preserved unless you explicitly change the name."
                    )
                    with st.form("update_existing_profile_form"):
                        updated_name = st.text_input(
                            "Update name (optional; leave blank to keep current name)",
                            value=selected_profile["name"],
                            max_chars=100,
                        )
                        description_addition = st.text_area(
                            "New description to add (optional)",
                            placeholder="Add new information; existing description will be preserved.",
                            max_chars=1000,
                        )
                        notes_addition = st.text_area(
                            "New notes to append (optional)",
                            placeholder="Add new notes; existing notes will be preserved.",
                            max_chars=3000,
                        )
                        confirm_link = st.checkbox(
                            "I reviewed the candidate and confirm this sample belongs to the selected profile.",
                            key="confirm_profile_link",
                        )
                        update_profile_submitted = st.form_submit_button(
                            "Confirm same person & update existing profile"
                        )

                    if update_profile_submitted:
                        if not selected_identity.get("quality", {}).get("quality_ok", False):
                            st.error("This photo did not pass the minimum quality check. Upload a clearer photo before adding it to a profile.")
                        elif not confirm_link:
                            st.error("Please confirm the same-person match before updating the profile.")
                        else:
                            try:
                                final_name = update_face_profile_info(
                                    db_engine,
                                    selected_profile_id,
                                    new_name=updated_name,
                                    description_addition=description_addition,
                                    notes_addition=notes_addition,
                                )
                                add_face_profile_sample(
                                    db_engine, selected_profile_id, selected_identity["embedding"],
                                    sample_fingerprint=current_image_fingerprint,
                                    sample_face_index=identity_index,
                                )
                                st.session_state["confirmed_profile_id_for_current_image"] = int(selected_profile_id)
                                st.session_state["confirmed_profile_image_fingerprint"] = current_image_fingerprint
                                st.session_state["confirmed_profile_face_index"] = matched_measurement_index
                                st.session_state["confirmed_profile_identity_index"] = identity_index
                                if matched_measurement_index is None:
                                    st.warning(
                                        "Profile information and face sample were updated, but the measurement detector "
                                        "could not confidently map this face to a row. The measurement record will not be linked until it can be mapped."
                                    )
                                else:
                                    st.success(
                                        f"Profile '{final_name}' updated. New sample added; new description/notes were appended. "
                                        "Click 'Save analysis to database' to link the measurement record."
                                    )
                                st.rerun()
                            except Exception:
                                st.error("Could not update the face profile. Check the private app logs for details; sensitive connection information is not shown here.")

                    st.divider()
                    st.markdown("**Delete this profile**")
                    st.caption(
                        "Deletion permanently removes the selected profile and its stored embeddings. "
                        "Historical measurement rows are retained but become unlinked."
                    )
                    confirm_delete = st.checkbox(
                        f"I understand that '{selected_profile['name']}' and its saved face samples will be deleted.",
                        key=f"confirm_delete_profile_{selected_profile_id}",
                    )
                    if st.button("Delete selected face profile", key="delete_face_profile"):
                        if not confirm_delete:
                            st.error("Confirm deletion first.")
                        else:
                            try:
                                delete_face_profile(db_engine, selected_profile_id)
                                st.session_state.pop("confirmed_profile_id_for_current_image", None)
                                st.session_state.pop("confirmed_profile_face_index", None)
                                st.session_state.pop("confirmed_profile_identity_index", None)
                                st.session_state.pop("confirmed_profile_image_fingerprint", None)
                                st.success("Face profile deleted.")
                                st.rerun()
                            except Exception:
                                st.error("Could not delete the face profile. Check the private app logs for details; sensitive connection information is not shown here.")

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
                        st.caption(
                            "No analysis records have been linked to this profile yet. Confirm a face sample, "
                            "then click 'Save analysis to database'."
                        )

            with st.expander("Create a new face profile", expanded=False):
                with st.form("create_face_profile_form"):
                    new_name = st.text_input("Name (required)", max_chars=100)
                    new_description = st.text_area("Description", max_chars=1000)
                    new_notes = st.text_area("Notes", max_chars=3000)
                    confirm_save = st.checkbox("I have permission to save and compare this face sample.")
                    create_submitted = st.form_submit_button("Save new face profile")
                if create_submitted:
                    if not selected_identity.get("quality", {}).get("quality_ok", False):
                        st.error("This photo did not pass the minimum quality check. Upload a clearer photo before creating a profile.")
                    elif not new_name.strip():
                        st.error("Enter a name before saving.")
                    elif not confirm_save:
                        st.error("Confirm that you have permission to process this face sample.")
                    else:
                        try:
                            profile_id = create_face_profile(
                                db_engine, new_name, new_description, new_notes,
                                selected_identity["embedding"],
                                sample_fingerprint=current_image_fingerprint,
                                sample_face_index=identity_index,
                            )
                            st.session_state["confirmed_profile_id_for_current_image"] = profile_id
                            st.session_state["confirmed_profile_image_fingerprint"] = current_image_fingerprint
                            st.session_state["confirmed_profile_face_index"] = matched_measurement_index
                            st.session_state["confirmed_profile_identity_index"] = identity_index
                            st.success(f"Profile '{new_name.strip()}' saved (ID {profile_id}).")
                            st.rerun()
                        except Exception:
                            st.error("Could not save the face profile. Check the private app logs for details; sensitive connection information is not shown here.")

    except Exception:
        st.warning(
            "Face profile matching is unavailable. Confirm that insightface and onnxruntime "
            "are installed and the model can load. Check the private app logs for details."
        )

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
    except Exception:
        st.warning("Could not load accumulated saved data. Check the private app logs for details.")
