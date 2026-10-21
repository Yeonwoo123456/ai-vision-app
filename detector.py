import cv2

print("OpenCV version:", getattr(cv2, "__version__", "unknown"))
print("OpenCV path:", getattr(cv2, "__file__", "unknown"))
print(
    "CascadeClassifier available:",
    hasattr(cv2, "CascadeClassifier")
)
import cv2
import numpy as np


def detect_faces(image_rgb):
    """
    RGB 이미지에서 얼굴 영역을 검출한다.
    반환값: (검출 결과 이미지, 얼굴 영역 좌표 목록)
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("유효한 이미지를 입력해야 합니다.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("얼굴 검출 모델을 불러오지 못했습니다.")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    result = image_rgb.copy()
    face_boxes = []

    for (x, y, w, h) in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        face_boxes.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })

    return result, face_boxes
