
import streamlit as st
import numpy as np
from PIL import Image
from detector import detect_faces

st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="🔬",
    layout="wide"
)

st.title("AI Vision Lab")
st.write("카메라 입력 및 얼굴 영역 검출")

photo = st.camera_input("카메라로 사진 촬영")

if photo is not None:
    image = Image.open(photo).convert("RGB")
    image_rgb = np.array(image)

    try:
        result, face_boxes = detect_faces(image_rgb)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("원본 이미지")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.subheader("검출 결과")
            st.image(result, use_container_width=True)

        st.metric("검출된 얼굴 영역 수", len(face_boxes))

        if face_boxes:
            st.dataframe(face_boxes, use_container_width=True)
        else:
            st.info("얼굴을 검출하지 못했습니다.")

    except Exception as error:
        st.error(f"이미지 처리 오류: {error}")

st.caption(
    "현재 버전은 얼굴 영역만 검출합니다. "
    "개인의 신원을 확인하거나 추정하지 않습니다."
)
