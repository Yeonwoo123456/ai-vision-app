import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sqlalchemy import create_engine, text
from google import genai
from detector import detect_faces
from ui import configure_page, render_header, render_sidebar, show_image


configure_page()



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
        # Idempotency key: the same uploaded image/face should not create duplicate history rows.
        connection.execute(text("ALTER TABLE analysis_history ADD COLUMN IF NOT EXISTS image_fingerprint TEXT NULL"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_analysis_image_face ON analysis_history (image_fingerprint, face_index) WHERE image_fingerprint IS NOT NULL"))
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
                created_at TEXT NOT NULL,
                sample_fingerprint TEXT NULL,
                sample_face_index INTEGER NULL
            )
        """))
        connection.execute(text("ALTER TABLE face_profile_embeddings ADD COLUMN IF NOT EXISTS sample_fingerprint TEXT NULL"))
        connection.execute(text("ALTER TABLE face_profile_embeddings ADD COLUMN IF NOT EXISTS sample_face_index INTEGER NULL"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_profile_sample_fingerprint ON face_profile_embeddings (profile_id, sample_fingerprint, sample_face_index) WHERE sample_fingerprint IS NOT NULL AND sample_face_index IS NOT NULL"))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS face_match_reviews (
                id BIGSERIAL PRIMARY KEY,
                profile_id BIGINT NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                image_fingerprint TEXT NOT NULL,
                decision TEXT NOT NULL,
                created_at TEXT NOT NULL,
                similarity_score DOUBLE PRECISION NULL,
                candidate_rank INTEGER NULL
            )
        """))
        # Safe migration for review records created by earlier app versions.
        connection.execute(text("ALTER TABLE face_match_reviews ADD COLUMN IF NOT EXISTS similarity_score DOUBLE PRECISION NULL"))
        connection.execute(text("ALTER TABLE face_match_reviews ADD COLUMN IF NOT EXISTS candidate_rank INTEGER NULL"))
        # Remove duplicate historical review events before enforcing idempotency going forward.
        connection.execute(text("""
            DELETE FROM face_match_reviews older
            USING face_match_reviews newer
            WHERE older.profile_id = newer.profile_id
              AND older.image_fingerprint = newer.image_fingerprint
              AND older.decision = newer.decision
              AND older.id < newer.id
        """))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_profile_review_event ON face_match_reviews (profile_id, image_fingerprint, decision)"))


def save_analysis(engine, face_count, width, height, result, profile_id=None, image_fingerprint=None):
    """Idempotently save a face analysis; repeated clicks do not duplicate the same image/face."""
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


def assess_face_quality(image_rgb, bbox, det_score, keypoints=None):
    """Estimate input quality from face size, detector confidence, blur and pose cues.

    These are screening heuristics, not a guarantee that a face match is correct.
    The source image is processed in memory and is not persisted by this function.
    """
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

    # Face size relative to the full image helps detect distant/tiny face crops.
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

    # Approximate pose cues from InsightFace's five keypoints (eyes, nose, mouth corners).
    # These are screening heuristics, not precise pose estimates.
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
                    # Eye keypoint order can run either left-to-right or right-to-left.
                    # Normalize orientation so a horizontal eye line is close to 0 degrees.
                    eye_roll_degrees = min(raw_eye_angle, abs(180.0 - raw_eye_angle))
                    if pose_offset > 0.22:
                        warnings.append("The face may be turned to the side; a more front-facing photo may match better.")
                    if eye_roll_degrees > 15:
                        warnings.append("The head appears tilted; a more level photo may improve matching.")
        except (TypeError, ValueError, IndexError):
            pose_offset = None
            eye_roll_degrees = None

    # A conservative quality gate. Warnings can appear before a hard block; quality is
    # only a photo-screening signal and does not establish identity.
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
    """Return local face embeddings and quality metadata; never saves the source image."""
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
                # Ignore malformed legacy samples rather than breaking all profile matching.
                continue
    return list(profiles.values())


def normalize_embedding(embedding):
    """Validate and L2-normalize an embedding before storing or comparing it."""
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
    """Update an existing profile; append new descriptive details without erasing prior notes."""
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

        # Add only new, non-duplicate text. Existing profile information is preserved.
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
        # Preserve the accumulated measurement history; remove only its profile link.
        connection.execute(text("UPDATE analysis_history SET profile_id = NULL WHERE profile_id = :id"), {"id": int(profile_id)})
        connection.execute(text("DELETE FROM face_match_reviews WHERE profile_id = :id"), {"id": int(profile_id)})
        connection.execute(text("DELETE FROM face_profiles WHERE id = :id"), {"id": int(profile_id)})


def find_face_profile_matches(embedding, profiles):
    """Rank profiles against multiple saved samples, not just a single best sample.

    The aggregate is a heuristic score, not an identity probability. Thresholds must be
    calibrated using consented, representative test photos before production use.
    """
    query = normalize_embedding(embedding)
    matches = []
    for profile in profiles:
        scores = []
        for sample in profile["embeddings"]:
            try:
                sample = normalize_embedding(sample)
            except ValueError:
                continue
            if sample.shape == query.shape:
                scores.append(float(np.dot(query, sample)))
        if scores:
            scores.sort(reverse=True)
            top_scores = scores[:min(3, len(scores))]
            best_score = top_scores[0]
            top_mean = float(np.mean(top_scores))
            # Blend best match with consistency across the closest few samples.
            aggregate_score = 0.65 * best_score + 0.35 * top_mean
            matches.append({
                **profile,
                "similarity": float(aggregate_score),
                "best_sample_similarity": float(best_score),
                "samples_compared": len(scores),
                "top_sample_mean": top_mean,
            })
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


def record_profile_review(
    engine, profile_id, image_fingerprint, decision="user_confirmed_same_person",
    similarity_score=None, candidate_rank=None,
):
    """Store a human-reviewed match label for later evaluation; not model training."""
    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO face_match_reviews (
                profile_id, image_fingerprint, decision, created_at, similarity_score, candidate_rank
            ) VALUES (
                :profile_id, :image_fingerprint, :decision, :created_at, :similarity_score, :candidate_rank
            )
            ON CONFLICT (profile_id, image_fingerprint, decision) DO NOTHING
        """), {
            "profile_id": int(profile_id),
            "image_fingerprint": image_fingerprint,
            "decision": decision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "similarity_score": float(similarity_score) if similarity_score is not None else None,
            "candidate_rank": int(candidate_rank) if candidate_rank is not None else None,
        })


def load_match_review_data(engine):
    """Load user-labeled match outcomes for evaluation and CSV export."""
    with engine.connect() as connection:
        rows = connection.execute(text("""
            SELECT r.id, r.created_at, r.profile_id,
                   COALESCE(NULLIF(p.name, ''), p.alias, '[deleted profile]') AS profile_name,
                   r.decision, r.similarity_score, r.candidate_rank, r.image_fingerprint
            FROM face_match_reviews r
            LEFT JOIN face_profiles p ON p.id = r.profile_id
            ORDER BY r.id DESC
        """)).mappings().all()
    return pd.DataFrame(rows)


def get_match_threshold():
    """Read the candidate threshold from Streamlit Secrets/env; default is a placeholder."""
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


def evaluate_reviewed_thresholds(review_df):
    """Evaluate candidate thresholds against human-labeled comparisons only."""
    if review_df is None or review_df.empty or "similarity_score" not in review_df.columns:
        return pd.DataFrame()
    labeled = review_df[review_df["decision"].isin([
        "user_confirmed_same_person", "user_confirmed_different_person"
    ])].copy()
    labeled["similarity_score"] = pd.to_numeric(labeled["similarity_score"], errors="coerce")
    labeled = labeled.dropna(subset=["similarity_score"])
    if labeled.empty:
        return pd.DataFrame()
    actual_same = (labeled["decision"] == "user_confirmed_same_person").to_numpy(dtype=bool)
    scores = labeled["similarity_score"].to_numpy(dtype=float)
    rows = []
    for threshold in np.round(np.arange(0.30, 0.851, 0.01), 2):
        predicted_same = scores >= threshold
        tp = int(np.sum(predicted_same & actual_same))
        fp = int(np.sum(predicted_same & ~actual_same))
        tn = int(np.sum(~predicted_same & ~actual_same))
        fn = int(np.sum(~predicted_same & actual_same))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced_accuracy = (recall + specificity) / 2 if (tp + fn) and (tn + fp) else None
        rows.append({
            "threshold": float(threshold), "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn, "precision": precision,
            "recall": recall, "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "labeled_comparisons": int(len(scores)),
        })
    return pd.DataFrame(rows)


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
    """Show saved-profile context only; do not call Gemini when no profile matches."""
    profile_context = profile_context or {}
    if profile_context.get("no_match") or profile_context.get("status") == "no saved profile match":
        return "No profile matching this person was found. If you would like, please create a new profile."

    api_key, model = get_gemini_settings()
    if not api_key:
        raise ValueError("Gemini API key is not configured. Add GEMINI_API_KEY to Streamlit Secrets.")

    # Deliberately provide only saved profile metadata to Gemini. Raw measurements,
    # measurement summaries, and limitation sections are not part of the visible AI output.
    profile_payload = {
        "saved_profile_name": profile_context.get("name"),
        "user_provided_description": profile_context.get("description", ""),
        "user_provided_notes": profile_context.get("notes", ""),
        "match_similarity_score": profile_context.get("similarity"),
        "profile_match_status": profile_context.get("status", "no confirmed profile match"),
    }

    prompt = (
        "You are the AI Vision Lab saved-profile context formatter. Output exactly these four lines and nothing else, "
        "with no heading, paragraph, bullets, markdown, or extra explanation:\n"
        "Name: <saved profile name or Not available>\n"
        "Description: <user-provided description or Not available>\n"
        "Note: <user-provided notes or Not available>\n"
        "Similarity: <model-specific score and match status, or Not available>\n\n"
        "For a possible match, format Similarity as '<score> (model-specific score, not a probability; confirm before linking)'. "
        "For a user-confirmed match, format Similarity as '<score if available> (user-confirmed)' or 'User-confirmed; score not available'. "
        "If no profile is available, use 'Not available' for Name, Description, and Note, and 'No saved profile match' for Similarity. "
        "Treat all profile text as user-provided information, not verified facts. Preserve the user's wording and do not invent details. "
        "Do not include measurement summaries, landmark values, image dimensions, limitations, or any other content. "
        "Do not infer identity or personal traits beyond the provided fields.\n\n"
        "Saved profile context (user-provided metadata):\n"
        + json.dumps(profile_payload, ensure_ascii=False, indent=2)
    )

    client = genai.Client(api_key=api_key)
    response = None
    last_error = None
    # Gemini 503 UNAVAILABLE is usually temporary. Retry briefly with backoff.
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            error_text = str(exc).upper()
            is_temporary = any(token in error_text for token in ("503", "UNAVAILABLE", "502", "504", "INTERNAL SERVER ERROR"))
            if not is_temporary or attempt == 2:
                raise
            time.sleep(1.5 * (2 ** attempt))
    if last_error is not None or response is None:
        raise RuntimeError(f"Gemini temporarily unavailable after retries: {last_error}")
    output = (response.text or "").strip()
    if not output:
        raise ValueError("Gemini returned an empty profile context. Please try again.")

    # Normalize to the exact four requested fields, even if Gemini adds extra prose.
    import re
    output = re.sub(r"(?im)^#+\s*Profile context\s*$", "", output).strip()
    field_values = {}
    for field in ("Name", "Description", "Note", "Similarity"):
        match = re.search(rf"(?im)^\s*{field}\s*:\s*(.*?)\s*$", output)
        field_values[field] = match.group(1).strip() if match else "Not available"
    if field_values["Similarity"] == "Not available" and profile_payload.get("match_similarity_score") is not None:
        score = profile_payload["match_similarity_score"]
        status = str(profile_payload.get("profile_match_status", ""))
        if "possible" in status.lower():
            field_values["Similarity"] = f"{score:.3f} (model-specific score, not a probability; confirm before linking)"
        else:
            field_values["Similarity"] = f"{score:.3f} (user-confirmed)"
    if "possible" in str(profile_payload.get("profile_match_status", "")).lower():
        if field_values["Similarity"] == "Not available":
            field_values["Similarity"] = "Not available (possible match; confirm before linking)"
        elif "confirm before linking" not in field_values["Similarity"].lower():
            field_values["Similarity"] += "; confirm before linking"
    return "\n\n".join(f"**{field}:** {field_values[field]}" for field in ("Name", "Description", "Note", "Similarity"))


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
        # Keep connection strings and provider diagnostics out of the public UI.
        st.warning("Database connection failed. Check DATABASE_URL in Streamlit Secrets and review the private app logs for details.")

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
                    explanation = generate_ai_measurement_explanation(
                        selected, original.shape[1], original.shape[0], profile_context
                    )
                st.session_state["ai_measurement_explanation"] = explanation
                st.session_state["ai_measurement_explanation_face"] = selected_index
            except Exception as exc:
                error_text = str(exc)
                if "503" in error_text or "UNAVAILABLE" in error_text.upper():
                    st.error(
                        "Gemini is temporarily busy (503 UNAVAILABLE). The app retried automatically, "
                        "but the model is still unavailable. Wait a little and try again, or select another "
                        "model available to your API key in Streamlit Secrets (GEMINI_MODEL). "
                        f"Details: {exc}"
                    )
                else:
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
            # Candidate profile details are intentionally hidden in this section.
            # Low-quality images are not ranked; ask for a clearer, more frontal photo.
            match_threshold = get_match_threshold()
            candidate = matches[0] if matches and matches[0]["similarity"] >= match_threshold else None
            if not quality.get("quality_ok"):
                st.info("Profile matching is paused for this image because quality is below the minimum screening threshold. Upload a clearer, larger, sharper, more front-facing photo.")
            elif profiles and not candidate:
                st.caption(f"No profile reached the current candidate threshold ({match_threshold:.2f}). This does not prove the person is new.")

            # They are shown only inside AI Explanation after the user requests it.

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

                # Explicit negative feedback helps measure false-match behavior.
                # It does not delete or modify the selected profile.
                st.caption("Accuracy feedback: use this only after reviewing the selected profile against the uploaded face.")
                if st.button(
                    "Mark selected profile as a different person",
                    key="mark_selected_profile_different_person",
                ):
                    if not quality.get("quality_ok", False):
                        st.error("The image quality is too low to label this comparison reliably.")
                    else:
                        selected_profile_match = next(
                            (m for m in matches if int(m["id"]) == int(selected_profile_id)), None
                        )
                        try:
                            record_profile_review(
                                db_engine, selected_profile_id, current_image_fingerprint,
                                decision="user_confirmed_different_person",
                                similarity_score=(selected_profile_match["similarity"] if selected_profile_match else None),
                                candidate_rank=next((i + 1 for i, m in enumerate(matches) if int(m["id"]) == int(selected_profile_id)), None),
                            )
                            st.success("Feedback saved: this uploaded face is a different person from the selected profile.")
                        except Exception:
                            st.error("Could not save evaluation feedback. Check the private app logs for details.")

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
                                selected_profile_match = next(
                                    (m for m in matches if int(m["id"]) == int(selected_profile_id)), None
                                )
                                record_profile_review(
                                    db_engine, selected_profile_id, current_image_fingerprint,
                                    decision="user_confirmed_same_person",
                                    similarity_score=(selected_profile_match["similarity"] if selected_profile_match else None),
                                    candidate_rank=next((i + 1 for i, m in enumerate(matches) if int(m["id"]) == int(selected_profile_id)), None),
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

            # New profile creation is intentionally placed below the update/delete controls.
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
                            record_profile_review(db_engine, profile_id, current_image_fingerprint, "user_created_profile")
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


# Human-reviewed matching evaluation. This summarizes labeled comparisons; it is not
# a statistically valid accuracy estimate unless the test set is representative.
if db_engine is not None:
    st.divider()
    st.subheader("Match Accuracy Evaluation")
    st.caption(
        "Review outcomes are human-provided labels, not automatic model training. "
        "Counts alone are not a true accuracy score unless you evaluate a representative, labeled test set."
    )
    try:
        review_df = load_match_review_data(db_engine)
        if review_df.empty:
            st.info("No reviewed match comparisons yet. Confirm same-person matches or mark a selected profile as a different person.")
        else:
            labeled = review_df[review_df["decision"].isin([
                "user_confirmed_same_person", "user_confirmed_different_person"
            ])].copy()
            same_count = int((labeled["decision"] == "user_confirmed_same_person").sum())
            different_count = int((labeled["decision"] == "user_confirmed_different_person").sum())
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Reviewed comparisons", len(labeled))
            metric_b.metric("Confirmed same person", same_count)
            metric_c.metric("Confirmed different person", different_count)
            if not labeled.empty:
                st.dataframe(
                    labeled[["created_at", "profile_name", "decision", "similarity_score", "candidate_rank"]].head(100),
                    use_container_width=True, hide_index=True
                )
            threshold_table = evaluate_reviewed_thresholds(labeled)
            if not threshold_table.empty:
                st.markdown("**Candidate threshold evaluation**")
                st.caption(
                    "This compares candidate thresholds against your human-reviewed labels. "
                    "It is only a preliminary estimate: repeated or non-representative examples can bias results. "
                    "The app does not automatically change its threshold."
                )
                current_threshold = get_match_threshold()
                has_both_classes = (
                    (labeled["decision"] == "user_confirmed_same_person").sum() >= 3
                    and (labeled["decision"] == "user_confirmed_different_person").sum() >= 3
                )
                if has_both_classes:
                    usable = threshold_table.dropna(subset=["balanced_accuracy"])
                    if not usable.empty:
                        best_row = usable.sort_values(
                            ["balanced_accuracy", "false_positive", "false_negative"],
                            ascending=[False, True, True],
                        ).iloc[0]
                        st.write(
                            f"Current configured threshold: **{current_threshold:.2f}**. "
                            f"Best threshold in this small reviewed sample: **{best_row['threshold']:.2f}** "
                            f"(balanced accuracy {best_row['balanced_accuracy']:.1%}). "
                            "Treat this as a starting point, not a validated production setting."
                        )
                        st.dataframe(
                            threshold_table[["threshold", "false_positive", "false_negative", "precision", "recall", "specificity", "balanced_accuracy"]].sort_values("threshold"),
                            use_container_width=True, hide_index=True,
                        )
                        st.caption("To test a candidate setting, add MATCH_THRESHOLD = \"0.50\" (replace 0.50 with your chosen value) to Streamlit Secrets, then redeploy.")
                else:
                    st.info(
                        "Collect at least 3 confirmed-same and 3 confirmed-different comparisons before using the suggested threshold as a rough guide. "
                        "More diverse, independently collected examples are better."
                    )

            st.download_button(
                "Download match evaluation CSV",
                data=review_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="face_match_evaluation.csv",
                mime="text/csv",
                key="download_match_evaluation_csv",
            )
    except Exception:
        st.warning("Could not load match evaluation data. Check the private app logs for details.")


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
    except Exception:
        st.warning("Could not load accumulated saved data. Check the private app logs for details.")
