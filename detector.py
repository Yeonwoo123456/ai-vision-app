
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
    """모델 파일이 없으면 다운로드한다."""
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


def draw_connections(image, landmarks, connections, color, thickness=1):
    """랜드마크 연결선을 이미지에 그린다."""
    height, width = image.shape[:2]

    for connection in connections:
        start = connection.start
        end = connection.end

        if start >= len(landmarks) or end >= len(landmarks):
            continue

        p1 = (
            int(landmarks[start].x * width),
            int(landmarks[start].y * height),
        )
        p2 = (
            int(landmarks[end].x * width),
            int(landmarks[end].y * height),
        )

        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


# 코 윤곽과 콧등을 강조하기 위한 랜드마크 연결선
# MediaPipe Face Landmarker의 표준 478개 랜드마크 인덱스 기준
NOSE_CONNECTIONS = [
    (168, 6), (6, 197), (197, 195), (195, 5),
    (5, 4), (4, 1), (1, 19), (19, 94),
    (94, 2), (2, 164), (164, 0),
    (98, 97), (97, 2), (2, 326), (326, 327),
    (129, 98), (98, 64), (64, 48),
    (358, 327), (327, 294), (294, 278),
    (48, 64), (64, 98), (98, 97),
    (278, 294), (294, 327), (327, 326),
]


def draw_nose_connections(image, landmarks, color, thickness=2):
    """코 랜드마크를 직접 연결한다."""
    height, width = image.shape[:2]

    for start, end in NOSE_CONNECTIONS:
        if start >= len(landmarks) or end >= len(landmarks):
            continue

        p1 = (
            int(landmarks[start].x * width),
            int(landmarks[start].y * height),
        )
        p2 = (
            int(landmarks[end].x * width),
            int(landmarks[end].y * height),
        )

        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def detect_faces(image_rgb):
    """얼굴 메시와 주요 부위 윤곽을 표시한다."""
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
        output_face_blendshapes=True,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb),
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    annotated_rgb = image_rgb.copy()
    face_data = []

    tessellation = (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    )

    contour_groups = [
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
            (255, 200, 0),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
            (255, 0, 0),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
            (255, 0, 0),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
            (255, 100, 0),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
            (255, 100, 0),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            (255, 255, 0),
            2,
        ),
    ]

    for face_index, landmarks in enumerate(result.face_landmarks):
        # 1. 얼굴 전체 메시
        draw_connections(
            annotated_rgb,
            landmarks,
            tessellation,
            color=(100, 220, 100),
            thickness=1,
        )

        # 2. 눈썹, 눈, 입, 얼굴 윤곽
        for connections, color, thickness in contour_groups:
            draw_connections(
                annotated_rgb,
                landmarks,
                connections,
                color=color,
                thickness=thickness,
            )

        # 3. 코 연결선
        draw_nose_connections(
            annotated_rgb,
            landmarks,
            color=(255, 0, 255),
            thickness=2,
        )

        # 4. 모든 랜드마크 점과 좌표
        height, width = annotated_rgb.shape[:2]
        points = []

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append({
                "x": round(float(landmark.x), 6),
                "y": round(float(landmark.y), 6),
                "z": round(float(landmark.z), 6),
            })

            cv2.circle(
                annotated_rgb,
                (x, y),
                1,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

        face_data.append({
            "face_index": face_index,
            "landmark_count": len(landmarks),
            "landmarks": points,
        })

    return annotated_rgb, face_data
