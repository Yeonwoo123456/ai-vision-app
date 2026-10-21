
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/"
    "face_landmarker.task"
)


def ensure_model():
    """MediaPipe 모델 파일이 없으면 다운로드한다."""
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

    if MODEL_PATH.stat().st_size == 0:
        raise RuntimeError("Face Landmarker 모델 파일이 비어 있습니다.")


def detect_faces(image_rgb):
    """
    RGB 이미지에서 얼굴 랜드마크를 검출한다.

    반환:
        annotated_rgb: 랜드마크가 그려진 RGB 이미지
        face_data: 얼굴별 랜드마크 좌표 목록
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("유효한 이미지를 입력해야 합니다.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("RGB 3채널 이미지를 입력해야 합니다.")

    ensure_model()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        output_face_blendshapes=True
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb)
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    annotated_rgb = image_rgb.copy()
    height, width = annotated_rgb.shape[:2]
    face_data = []

    connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

    for face_index, landmarks in enumerate(result.face_landmarks):
        points = []

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        # 얼굴 메시를 촘촘한 선으로 표시
        for connection in connections:
            start = connection.start
            end = connection.end

            if start < len(points) and end < len(points):
                cv2.line(
                    annotated_rgb,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        face_data.append({
            "face_index": face_index,
            "landmark_count": len(landmarks),
            "landmarks": [
                {
                    "x": round(float(point.x), 6),
                    "y": round(float(point.y), 6),
                    "z": round(float(point.z), 6)
                }
                for point in landmarks
            ]
        })

    return annotated_rgb, face_data
