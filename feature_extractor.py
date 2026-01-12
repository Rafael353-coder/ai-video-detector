import cv2
import numpy as np
import mediapipe as mp
from scipy.stats import entropy

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

def compute_features(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo")

    face_variances = []
    face_entropies = []
    face_movements = []
    global_movements = []

    prev_face = None
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Movimento global
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            global_movements.append(np.mean(diff))
        prev_gray = gray

        # Face detection (frame em memória ✔)
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box

            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            face = gray[y1:y2, x1:x2]

            if face.size > 0:
                face_variances.append(np.var(face))

                hist = cv2.calcHist([face], [0], None, [256], [0, 256]).flatten()
                face_entropies.append(entropy(hist + 1e-6))

                if prev_face is not None and prev_face.shape == face.shape:
                    face_movements.append(np.mean(cv2.absdiff(face, prev_face)))

                prev_face = face

    cap.release()

    return {
        "face_variance": float(np.mean(face_variances)) if face_variances else 0.0,
        "face_entropy": float(np.mean(face_entropies)) if face_entropies else 0.0,
        "face_temporal": float(np.mean(face_movements)) if face_movements else 0.0,
        "global_temporal": float(np.mean(global_movements)) if global_movements else 0.0,
    }

