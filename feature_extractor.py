import cv2
import numpy as np
import mediapipe as mp


# -------------------------
# Inicialização MediaPipe (API PÚBLICA)
# -------------------------
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# -------------------------
# Funções auxiliares
# -------------------------
def compute_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-6)
    return -np.sum(hist * np.log2(hist + 1e-6))


# -------------------------
# Feature extractor principal
# -------------------------
def compute_features(video_path):
    cap = cv2.VideoCapture(video_path)

    face_variances = []
    face_entropies = []
    face_temporals = []
    global_temporals = []

    prev_face = None
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------
        # Movimento global
        # -------------------------
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            global_temporals.append(np.mean(diff))

        prev_gray = gray

        # -------------------------
        # Deteção facial
        # -------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        if results.detections:
            box = 
results.detections[0].location_data.relative_bounding_box

            h, w, _ = frame.shape
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            # Segurança de limites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = gray[y1:y2, x1:x2]

            if face.size > 0:
                face_variances.append(np.var(face))
                face_entropies.append(compute_entropy(face))

                if prev_face is not None and prev_face.shape == 
face.shape:
                    face_temporals.append(np.mean(np.abs(face - 
prev_face)))

                prev_face = face

    cap.release()

    # -------------------------
    # Resultados finais
    # -------------------------
    return {
        "face_variance": float(np.mean(face_variances)) if face_variances 
else 0.0,
        "face_entropy": float(np.mean(face_entropies)) if face_entropies 
else 0.0,
        "face_temporal": float(np.mean(face_temporals)) if face_temporals 
else 0.0,
        "global_temporal": float(np.mean(global_temporals)) if 
global_temporals else 0.0,
    }

