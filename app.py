
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from detector import detect_faces

st.set_page_config(page_title="AI Vision Lab", layout="wide")
st.title("AI Vision Lab")
st.write("MediaPipe Face Landmarker 테스트")

photo = st.camera_input("사진 촬영")

if photo is not None:
    image_rgb = np.array(Image.open(photo).convert("RGB"))

    try:
        with st.spinner("얼굴 랜드마크 분석 중..."):
            result_image, faces = detect_faces(image_rgb)

        left, right = st.columns(2)

        with left:
            st.subheader("원본")
            st.image(image_rgb, use_container_width=True)

        with right:
            st.subheader("얼굴 메시")
            st.image(result_image, use_container_width=True)

        st.metric("검출된 얼굴 수", len(faces))

        if faces:
            selected_face = st.selectbox(
                "확인할 얼굴",
                options=range(len(faces)),
                format_func=lambda i: f"Face {i + 1}"
            )

            face = faces[selected_face]
            st.metric("랜드마크 수", face["landmark_count"])

            df = pd.DataFrame(face["landmarks"])
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "좌표 CSV 다운로드",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="face_landmarks.csv",
                mime="text/csv"
            )
        else:
            st.info("얼굴을 검출하지 못했습니다. 정면에서 다시 촬영해 보세요.")

    except Exception as e:
        st.error(f"이미지 처리 오류: {e}")
