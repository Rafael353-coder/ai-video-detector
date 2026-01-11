import cv2
import numpy as np
import mediapipe
from mediapipe.python.solutions import face_detection
import tempfile
import os


# -------------------------------------------------
# Inicialização do MediaPipe (FORMA COMPATÍVEL)
# -------------------------------------------------
mp_face = face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# -------------------------------------------------
# Funções auxiliares
# -------------------------------------------------
def entropy(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-6)
    return -np.sum(hist * np.log2(hist + 1e-6))


def resize_face(face_img, size=(128, 128)):
    return cv2.resize(face_img, size, interpolation=cv2.INTER_LINEAR)


# -------------------------------------------------
# Função principal
# -------------------------------------------------
def compute_features(video_path):
    cap = cv2.VideoCapture(video_path)

    face_frames = []
    prev_face = None
    face_temporals = []
    global_temporals = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Movimento global
        if prev_gray is not None:
            global_temporals.append(np.mean(np.abs(gray - prev_gray)))
        prev_gray = gray

        # Detecção facial
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            if x2 > x1 and y2 > y1:
                face = gray[y1:y2, x1:x2]
                face = resize_face(face)

                face_frames.append(face)

                if prev_face is not None:
                    face_temporals.append(
                        np.mean(np.abs(face.astype(np.float32) - 
prev_face.astype(np.float32)))
                    )

                prev_face = face

    cap.release()

    # -------------------------------------------------
    # Extração das métricas
    # -------------------------------------------------
    if len(face_frames) == 0:
        return {
            "face_variance": 0.0,
            "face_entropy": 0.0,
            "face_temporal": 0.0,
            "global_temporal": float(np.mean(global_temporals)) if 
global_temporals else 0.0
        }

    face_stack = np.stack(face_frames)

    return {
        "face_variance": float(np.var(face_stack)),
        "face_entropy": float(np.mean([entropy(f) for f in face_frames])),
        "face_temporal": float(np.mean(face_temporals)) if face_temporals 
else 0.0,
        "global_temporal": float(np.mean(global_temporals)) if 
global_temporals else 0.0
    }


# -------------------------------------------------
# Execução direta (teste local)
# -------------------------------------------------
if __name__ == "__main__":
    import sys
    print(compute_features(sys.argv[1]))

