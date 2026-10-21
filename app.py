
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from detector import detect_faces


st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Vision Lab")
st.caption("Computer Vision · Facial Landmark Analysis · Analytics Dashboard")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("이미지 입력 방식을 선택하세요.")

    input_mode = st.radio(
        "Image source",
        ["Camera", "Upload image"],
    )

    show_landmarks = st.checkbox(
        "Show landmark visualization",
        value=True,
    )

    show_coordinate_table = st.checkbox(
        "Show landmark coordinates",
        value=False,
    )

    st.divider()
    st.caption("분석 결과는 이미지 기반 상대 좌표입니다.")


# -----------------------------
# Image input
# -----------------------------
image_file = None

if input_mode == "Camera":
    image_file = st.camera_input("Take a picture")
else:
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

if image_file is None:
    st.info(
        "카메라로 사진을 촬영하거나 이미지를 업로드하면 "
        "얼굴 분석 대시보드가 표시됩니다."
    )

    st.markdown("### Analysis pipeline")
    st.code(
        "Image Input → Face Detection → Landmark Analysis "
        "→ Dashboard",
        language="text",
    )
    st.stop()


# -----------------------------
# Load image
# -----------------------------
try:
    image_bytes = image_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.array(pil_image)

except Exception as exc:
    st.error(f"이미지를 읽을 수 없습니다: {exc}")
    st.stop()


# -----------------------------
# Analysis helpers
# -----------------------------
def point_distance(p1, p2):
    """정규화된 좌표 두 점 사이의 유클리드 거리."""
    return float(
        np.sqrt(
            (p1["x"] - p2["x"]) ** 2
            + (p1["y"] - p2["y"]) ** 2
        )
    )


def center_of_points(landmarks, indices):
    """여러 랜드마크의 평균 좌표."""
    valid_points = [
        landmarks[i]
        for i in indices
        if i < len(landmarks)
    ]

    if not valid_points:
        return None

    return {
        "x": float(np.mean([p["x"] for p in valid_points])),
        "y": float(np.mean([p["y"] for p in valid_points])),
        "z": float(np.mean([p["z"] for p in valid_points])),
    }


def analyze_face(face):
    """랜드마크에서 상대 거리와 비율을 계산한다."""
    landmarks = face["landmarks"]

    def get_point(index):
        if index < len(landmarks):
            return landmarks[index]
        return None

    # MediaPipe Face Landmarker standard landmark indices
    left_eye = center_of_points(landmarks, [33, 133, 159, 145])
    right_eye = center_of_points(landmarks, [362, 263, 386, 374])

    nose_tip = get_point(1)
    upper_lip = get_point(13)
    lower_lip = get_point(14)

    face_left = get_point(234)
    face_right = get_point(454)
    face_top = get_point(10)
    face_bottom = get_point(152)

    eye_distance = None
    face_width = None
    face_height = None
    width_height_ratio = None
    nose_offset = None
    mouth_gap = None

    if left_eye and right_eye:
        eye_distance = point_distance(left_eye, right_eye)

    if face_left and face_right:
        face_width = point_distance(face_left, face_right)

    if face_top and face_bottom:
        face_height = point_distance(face_top, face_bottom)

    if face_width is not None and face_height is not None:
        if face_height > 0:
            width_height_ratio = face_width / face_height

    if left_eye and right_eye and nose_tip:
        eye_midpoint = {
            "x": (left_eye["x"] + right_eye["x"]) / 2,
            "y": (left_eye["y"] + right_eye["y"]) / 2,
        }

        nose_offset = {
            "x": nose_tip["x"] - eye_midpoint["x"],
            "y": nose_tip["y"] - eye_midpoint["y"],
        }

    if upper_lip and lower_lip:
        mouth_gap = point_distance(upper_lip, lower_lip)

    return {
        "face_index": face["face_index"],
        "landmark_count": face["landmark_count"],
        "eye_distance": eye_distance,
        "face_width": face_width,
        "face_height": face_height,
        "width_height_ratio": width_height_ratio,
        "nose_offset": nose_offset,
        "mouth_gap": mouth_gap,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "nose_tip": nose_tip,
    }


def relative_value(value):
    """정규화된 거리를 백분율로 표시한다."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


# -----------------------------
# Run face detection
# -----------------------------
with st.spinner("얼굴 감지 및 랜드마크 분석 중..."):
    try:
        annotated_rgb, face_data = detect_faces(original_rgb)
    except Exception as exc:
        st.error("이미지 처리 중 오류가 발생했습니다.")
        st.exception(exc)
        st.stop()


# -----------------------------
# Dashboard header
# -----------------------------
analyses = [analyze_face(face) for face in face_data]

face_count = len(face_data)
total_landmarks = sum(
    face["landmark_count"] for face in face_data
)

image_height, image_width = original_rgb.shape[:2]

st.subheader("📊 Analysis Overview")

metric_cols = st.columns(4)

metric_cols[0].metric("Faces detected", face_count)
metric_cols[1].metric("Total landmarks", total_landmarks)
metric_cols[2].metric("Image width", f"{image_width}px")
metric_cols[3].metric("Image height", f"{image_height}px")

if face_count == 0:
    st.warning(
        "얼굴을 감지하지 못했습니다. 얼굴이 잘 보이도록 "
        "조명을 밝게 하거나 다른 이미지를 사용해 보세요."
    )
    st.image(original_rgb, caption="Original image", use_container_width=True)
    st.stop()


# -----------------------------
# Image comparison
# -----------------------------
st.subheader("🖼️ Visual Analysis")

image_col1, image_col2 = st.columns(2)

with image_col1:
    st.markdown("**Original Image**")
    st.image(
        original_rgb,
        use_container_width=True,
    )

with image_col2:
    st.markdown("**Landmark Visualization**")

    if show_landmarks:
        st.image(
            annotated_rgb,
            use_container_width=True,
        )
    else:
        st.info("사이드바에서 랜드마크 시각화를 활성화하세요.")


# -----------------------------
# Per-face measurements
# -----------------------------
st.subheader("📐 Facial Feature Measurements")

selected_face_index = st.selectbox(
    "Select a detected face",
    options=list(range(face_count)),
    format_func=lambda i: f"Face {i + 1}",
)

selected = analyses[selected_face_index]

st.caption(
    "거리와 비율은 이미지의 정규화된 좌표를 기준으로 계산합니다. "
    "실제 신체 치수나 신원 확인 결과가 아닙니다."
)

measure_cols = st.columns(4)

measure_cols[0].metric(
    "Eye center distance",
    relative_value(selected["eye_distance"]),
)

measure_cols[1].metric(
    "Face width",
    relative_value(selected["face_width"]),
)

measure_cols[2].metric(
    "Face height",
    relative_value(selected["face_height"]),
)

ratio = selected["width_height_ratio"]
measure_cols[3].metric(
    "Width / height",
    f"{ratio:.3f}" if ratio is not None else "N/A",
)

extra_cols = st.columns(2)

extra_cols[0].metric(
    "Mouth opening distance",
    relative_value(selected["mouth_gap"]),
)

nose_offset = selected["nose_offset"]
if nose_offset is not None:
    nose_offset_text = (
        f"x: {nose_offset['x']:+.4f}, "
        f"y: {nose_offset['y']:+.4f}"
    )
else:
    nose_offset_text = "N/A"

extra_cols[1].metric(
    "Nose offset from eye midpoint",
    nose_offset_text,
)


# -----------------------------
# Per-face summary table
# -----------------------------
st.subheader("📋 Face Summary")

summary_rows = []

for result in analyses:
    summary_rows.append({
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
        "Width/height ratio": (
            round(result["width_height_ratio"], 3)
            if result["width_height_ratio"] is not None else None
        ),
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# Landmark coordinate details
# -----------------------------
if show_coordinate_table:
    st.subheader("🔎 Landmark Coordinates")

    selected_landmarks = face_data[selected_face_index]["landmarks"]

    coordinates_df = pd.DataFrame(selected_landmarks)
    coordinates_df.insert(
        0,
        "landmark_index",
        range(len(coordinates_df)),
    )

    st.dataframe(
        coordinates_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = coordinates_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download landmark coordinates (CSV)",
        data=csv_data,
        file_name=f"face_{selected_face_index + 1}_landmarks.csv",
        mime="text/csv",
    )


# -----------------------------
# Export summary
# -----------------------------
st.subheader("📥 Export Analysis")

summary_csv = summary_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download face analysis summary (CSV)",
    data=summary_csv,
    file_name="face_analysis_summary.csv",
    mime="text/csv",
)

st.caption(
    "AI Vision Lab · Facial landmarks are geometric estimates. "
    "Results may vary with lighting, pose, image quality, and camera angle."
)
