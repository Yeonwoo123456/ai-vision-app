
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


NOSE_CONNECTIONS = [
    (168, 6), (6, 197), (197, 195), (195, 5),
    (5, 4), (4, 1), (1, 19), (19, 94),
    (94, 2), (2, 164), (164, 0),
    (98, 97), (97, 2), (2, 326), (326, 327),
    (129, 98), (98, 64), (64, 48),
    (358, 327), (327, 294), (294, 278),
]


def ensure_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


def draw_connections(image, landmarks, connections, color, thickness=1):
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


def draw_nose(image, landmarks, color=(255, 0, 255), thickness=2):
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


def draw_point(image, landmark, color, radius=3):
    height, width = image.shape[:2]

    point = (
        int(landmark.x * width),
        int(landmark.y * height),
    )

    cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)


def draw_measurement(image, landmarks):
    height, width = image.shape[:2]

    if len(landmarks) <= 454:
        return

    left_eye_ids = [33, 133, 159, 145]
    right_eye_ids = [362, 263, 386, 374]

    def center(indices):
        points = np.array([
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in indices
        ])

        return tuple(np.round(points.mean(axis=0)).astype(int))

    left_eye = center(left_eye_ids)
    right_eye = center(right_eye_ids)
    face_left = (
        int(landmarks[234].x * width),
        int(landmarks[234].y * height),
    )
    face_right = (
        int(landmarks[454].x * width),
        int(landmarks[454].y * height),
    )
    face_top = (
        int(landmarks[10].x * width),
        int(landmarks[10].y * height),
    )
    face_bottom = (
        int(landmarks[152].x * width),
        int(landmarks[152].y * height),
    )

    cv2.line(image, left_eye, right_eye, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, face_left, face_right, (255, 255, 0), 2, cv2.LINE_AA)

    center_x = (face_top[0] + face_bottom[0]) // 2

    cv2.line(
        image,
        (center_x, face_top[1]),
        (center_x, face_bottom[1]),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for point in (left_eye, right_eye):
        cv2.circle(image, point, 5, (0, 255, 255), -1, cv2.LINE_AA)


def detect_faces(image_rgb):
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("A valid image is required.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("The image must have three RGB channels.")

    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    ensure_model()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        output_face_blendshapes=False,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    annotated_rgb = image_rgb.copy()
    face_data = []

    connections = [
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
            (0, 255, 255),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
            (255, 80, 0),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
            (255, 80, 0),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
            (0, 180, 255),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
            (0, 180, 255),
            2,
        ),
        (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            (0, 255, 0),
            2,
        ),
    ]

    for face_index, landmarks in enumerate(result.face_landmarks):
        draw_connections(
            annotated_rgb,
            landmarks,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            (100, 160, 100),
            1,
        )

        for group, color, thickness in connections:
            draw_connections(
                annotated_rgb,
                landmarks,
                group,
                color,
                thickness,
            )

        draw_nose(annotated_rgb, landmarks)
        draw_measurement(annotated_rgb, landmarks)

        points = []

        for landmark in landmarks:
            points.append({
                "x": round(float(landmark.x), 6),
                "y": round(float(landmark.y), 6),
                "z": round(float(landmark.z), 6),
            })

        face_data.append({
            "face_index": face_index,
            "landmark_count": len(points),
            "landmarks": points,
        })

    return annotated_rgb, face_data
