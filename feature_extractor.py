import os
import cv2
import numpy as np
import tempfile
import shutil
import subprocess
import mediapipe as mp

# =========================
# MediaPipe Face Detector
# =========================
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# =========================
# Extract frames with ffmpeg
# =========================
def extract_frames(video_path, frames_dir):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-t", "10",
            "-vf", "fps=1",
            os.path.join(frames_dir, "frame_%03d.jpg")
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# =========================
# Extract face from frame
# =========================
def extract_face(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if not results.detections:
        return None

    box = results.detections[0].location_data.relative_bounding_box
    h, w, _ = img.shape

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)

    face = img[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (128, 128))

    return face

# =========================
# Compute features
# =========================
def compute_features(video_path):
    tmp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(tmp_dir, "frames")
    os.mkdir(frames_dir)

    extract_frames(video_path, frames_dir)

    face_variances = []
    face_entropies = []
    face_temporals = []
    global_temporals = []

    prev_face = None
    prev_gray = None

    for f in sorted(os.listdir(frames_dir)):
        img = cv2.imread(os.path.join(frames_dir, f))
        if img is None:
            continue

        # -------- GLOBAL TEMPORAL --------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        if prev_gray is not None:
            global_temporals.append(
                np.mean(np.abs(gray - prev_gray))
            )

        prev_gray = gray

        # -------- FACE FEATURES --------
        face = extract_face(img)
        if face is None:
            continue

        face_variances.append(np.var(face))

        hist = cv2.calcHist([face], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-10)
        face_entropies.append(
            -np.sum(hist * np.log2(hist + 1e-10))
        )

        if prev_face is not None:
            face_temporals.append(
                np.mean(np.abs(face - prev_face))
            )

        prev_face = face

    shutil.rmtree(tmp_dir)

    return {
        "face_variance": float(np.mean(face_variances)) if face_variances 
else 0.0,
        "face_entropy": float(np.mean(face_entropies)) if face_entropies 
else 0.0,
        "face_temporal": float(np.mean(face_temporals)) if face_temporals 
else 0.0,
        "global_temporal": float(np.mean(global_temporals)) if 
global_temporals else 0.0
    }

# =========================
# CLI entry point
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extractor.py <video_path>")
    else:
        print(compute_features(sys.argv[1]))

