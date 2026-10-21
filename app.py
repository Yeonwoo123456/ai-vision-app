
import base64
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from detector import detect_faces


st.set_page_config(
    page_title="AI Vision Lab",
    layout="wide",
)


def load_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()

    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.load()
        result = np.array(image, dtype=np.uint8)

    if result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("The image must have three RGB channels.")

    if result.size == 0:
        raise ValueError("The image contains no pixel data.")

    return np.ascontiguousarray(result)


def validate_image(image):
    if image is None:
        raise ValueError("No image is available.")

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The image must have three RGB channels.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("The image dimensions are invalid.")

    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def show_image(image, caption):
    image = validate_image(image)

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    st.markdown(f"**{caption}**")

    st.html(
        f"""
        <div style="width:100%; overflow:hidden;">
            <img
                src="data:image/png;base64,{encoded}"
                alt="Image preview"
                style="
                    display:block;
                    width:100%;
                    height:auto;
                    object-fit:contain;
                "
            />
        </div>
        """
    )


def point_distance(p1, p2):
    return float(
        np.sqrt(
            (p1["x"] - p2["x"]) ** 2
            + (p1["y"] - p2["y"]) ** 2
        )
    )


def center_of_points(landmarks, indices):
    points = [
        landmarks[index]
        for index in indices
        if index < len(landmarks)
    ]

    if not points:
        return None

    return {
        "x": float(np.mean([point["x"] for point in points])),
        "y": float(np.mean([point["y"] for point in points])),
        "z": float(np.mean([point["z"] for point in points])),
    }


def analyze_face(face):
    landmarks = face["landmarks"]

    def get_point(index):
        if index < len(landmarks):
            return landmarks[index]
        return None

    left_eye = center_of_points(
        landmarks, [33, 133, 159, 145]
    )
    right_eye = center_of_points(
        landmarks, [362, 263, 386, 374]
    )

    nose_tip = get_point(1)
    upper_lip = get_point(13)
    lower_lip = get_point(14)

    face_left = get_point(234)
    face_right = get_point(454)
    face_top = get_point(10)
    face_bottom = get_point(152)

    eye_distance = (
        point_distance(left_eye, right_eye)
        if left_eye and right_eye else None
    )

    face_width = (
        point_distance(face_left, face_right)
        if face_left and face_right else None
    )

    face_height = (
        point_distance(face_top, face_bottom)
        if face_top and face_bottom else None
    )

    width_height_ratio = (
        face_width / face_height
        if face_width is not None
        and face_height is not None
        and face_height > 0
        else None
    )

    nose_offset = None

    if left_eye and right_eye and nose_tip:
        nose_offset = {
            "x": nose_tip["x"]
            - (left_eye["x"] + right_eye["x"]) / 2,
            "y": nose_tip["y"]
            - (left_eye["y"] + right_eye["y"]) / 2,
        }

    mouth_gap = (
        point_distance(upper_lip, lower_lip)
        if upper_lip and lower_lip else None
    )

    return {
        "face_index": face["face_index"],
        "landmark_count": face["landmark_count"],
        "eye_distance": eye_distance,
        "face_width": face_width,
        "face_height": face_height,
        "width_height_ratio": width_height_ratio,
        "nose_offset": nose_offset,
        "mouth_gap": mouth_gap,
    }


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


st.title("AI Vision Lab")
st.caption("Facial Landmark Analysis and Dashboard")

with st.sidebar:
    st.header("Settings")

    input_mode = st.radio(
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

if input_mode == "Camera":
    uploaded_file = st.camera_input("Take a picture")
else:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

if uploaded_file is None:
    st.info("Take a picture or upload an image to begin.")
    st.stop()

try:
    original_rgb = load_image(uploaded_file)
except Exception as exc:
    st.error(f"Unable to load image: {exc}")
    st.stop()

try:
    with st.spinner("Analyzing image..."):
        annotated_rgb, face_data = detect_faces(original_rgb)

    original_rgb = validate_image(original_rgb)
    annotated_rgb = validate_image(annotated_rgb)

    if annotated_rgb.shape != original_rgb.shape:
        raise ValueError(
            "The processed image dimensions do not match "
            "the original image."
        )

except Exception as exc:
    st.error(f"Image processing failed: {exc}")
    st.stop()

analyses = [analyze_face(face) for face in face_data]

st.subheader("Analysis Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Faces detected", len(face_data))
col2.metric(
    "Total landmarks",
    sum(face["landmark_count"] for face in face_data),
)
col3.metric("Image width", f"{original_rgb.shape[1]} px")
col4.metric("Image height", f"{original_rgb.shape[0]} px")

st.subheader("Visual Analysis")

image_col1, image_col2 = st.columns(2)

with image_col1:
    show_image(original_rgb, "Original Image")

with image_col2:
    if show_landmarks:
        show_image(annotated_rgb, "Landmark Visualization")
    else:
        st.info("Enable landmark visualization in Settings.")

if not face_data:
    st.warning(
        "No face was detected. Try an image with a clearly visible face."
    )
    st.stop()

st.subheader("Facial Feature Measurements")

selected_index = st.selectbox(
    "Select face",
    options=list(range(len(face_data))),
    format_func=lambda index: f"Face {index + 1}",
)

selected = analyses[selected_index]

st.caption(
    "Measurements are relative image-coordinate estimates, "
    "not physical dimensions or identity verification."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Eye center distance",
    format_percent(selected["eye_distance"]),
)
m2.metric(
    "Face width",
    format_percent(selected["face_width"]),
)
m3.metric(
    "Face height",
    format_percent(selected["face_height"]),
)
m4.metric(
    "Width / height ratio",
    f'{selected["width_height_ratio"]:.3f}'
    if selected["width_height_ratio"] is not None
    else "N/A",
)

m5, m6 = st.columns(2)

m5.metric(
    "Mouth opening distance",
    format_percent(selected["mouth_gap"]),
)

offset = selected["nose_offset"]

m6.metric(
    "Nose offset from eye midpoint",
    f'x: {offset["x"]:+.4f}, y: {offset["y"]:+.4f}'
    if offset else "N/A",
)

st.subheader("Face Summary")

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

st.subheader("Export Analysis")

st.download_button(
    "Download analysis summary CSV",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="face_analysis_summary.csv",
    mime="text/csv",
)

if show_coordinates:
    st.subheader("Landmark Coordinates")

    landmarks_df = pd.DataFrame(
        face_data[selected_index]["landmarks"]
    )

    landmarks_df.insert(
        0,
        "landmark_index",
        range(len(landmarks_df)),
    )

    st.dataframe(
        landmarks_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download landmark coordinates CSV",
        data=landmarks_df.to_csv(index=False).encode("utf-8"),
        file_name=f"face_{selected_index + 1}_landmarks.csv",
        mime="text/csv",
    )
