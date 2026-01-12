import cv2
import numpy as np
import os

# 🔐 MediaPipe DESATIVADO em produção (Render)
USE_MEDIAPIPE = os.getenv("RENDER") is None


def global_motion_only(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    motions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motions.append(np.mean(diff))

        prev_gray = gray

    cap.release()
    return float(np.mean(motions)) if motions else 0.0


def compute_features(video_path: str) -> dict:
    # 🚨 PRODUÇÃO (Render) — SEM MediaPipe
    if not USE_MEDIAPIPE:
        return {
            "face_variance": 0.0,
            "face_entropy": 0.0,
            "face_temporal": 0.0,
            "global_temporal": global_motion_only(video_path),
        }

    # 🧠 LOCAL (com MediaPipe)
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )

    face_variances = []
    entropies = []
    temporal = []

    prev_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        if results.detections:
            det = results.detections[0]
            box = det.location_data.relative_bounding_box

            h, w, _ = frame.shape
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_variances.append(np.var(gray))

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / (hist.sum() + 1e-6)
            entropies.append(-np.sum(hist * np.log2(hist + 1e-6)))

            if prev_face is not None and prev_face.shape == gray.shape:
                diff = cv2.absdiff(gray, prev_face)
                temporal.append(np.mean(diff))

            prev_face = gray

    cap.release()

    return {
        "face_variance": float(np.mean(face_variances)) if face_variances else 0.0,
        "face_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "face_temporal": float(np.mean(temporal)) if temporal else 0.0,
        "global_temporal": global_motion_only(video_path),
    }

