from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import math
import joblib
import pandas as pd

from image_features import compute_image_features
from feature_extractor import compute_features, extract_video_frames_for_image_model

IMAGE_MODEL_PATH = "model_image.pkl"
VIDEO_MODEL_PATH = "model_video.pkl"

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sanitize_dict(data: dict) -> dict:
    clean = {}
    for k, v in data.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                clean[k] = 0.0
            else:
                clean[k] = v
        else:
            clean[k] = v
    return clean


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo nao encontrado: {path}")
    return joblib.load(path)


def explain_features(features: dict) -> list:
    reasons = []

    if features.get("noise", 999) < 20:
        reasons.append("Ruido residual demasiado limpo")

    if features.get("edge_strength", 999) < 15:
        reasons.append("Textura excessivamente suave")

    if features.get("high_freq", 999) < 120:
        reasons.append("Pouca energia de alta frequencia")

    if features.get("laplacian", 999) < 100:
        reasons.append("Detalhe anormalmente baixo")

    if features.get("temporal_diff_mean", 999) < 8:
        reasons.append("Consistencia temporal artificial")

    if features.get("noise_level_mean", 999) < 5:
        reasons.append("Ruido temporal demasiado estavel")

    if not reasons:
        reasons.append("Sem sinais fortes de artificio")

    return reasons


def build_feature_row(features: dict, model) -> pd.DataFrame:
    X = pd.DataFrame([dict(features)])

    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]

    return X


def predict_image_features(features: dict):
    model = load_model(IMAGE_MODEL_PATH)
    X = build_feature_row(features, model)
    pred = int(model.predict(X)[0])
    prob_ai = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
    return pred, prob_ai


def predict_video_features(features: dict):
    model = load_model(VIDEO_MODEL_PATH)
    X = build_feature_row(features, model)
    pred = int(model.predict(X)[0])
    prob_ai = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
    return pred, prob_ai


def predict_video_with_hybrid_strategy(video_path: str):
    video_features = sanitize_dict(compute_features(video_path))
    video_pred, video_prob = predict_video_features(video_features)

    frame_features_list = extract_video_frames_for_image_model(video_path, max_frames=8)

    image_frame_probs = []
    for feats in frame_features_list:
        feats = sanitize_dict(feats)
        _, prob = predict_image_features(feats)
        image_frame_probs.append(prob)

    image_prob_mean = sum(image_frame_probs) / len(image_frame_probs) if image_frame_probs else 0.0

    final_prob = (0.25 * video_prob) + (0.75 * image_prob_mean)

    risk = max(0, min(100, int(final_prob * 100)))

    if risk < 30:
        level = "BAIXO"
    elif risk < 60:
        level = "MEDIO"
    elif risk < 80:
        level = "SUSPEITO"
    else:
        level = "ALTO"

    return {
        "risk": risk,
        "level": level,
        "ml_prediction": 1 if final_prob >= 0.5 else 0,
        "video_model_prob": round(video_prob, 4),
        "image_frame_prob_mean": round(image_prob_mean, 4),
        "reasons": explain_features(video_features),
        "features": video_features,
    }


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        features = sanitize_dict(compute_image_features(temp_path))
        pred, prob_ai = predict_image_features(features)

        risk = max(0, min(100, int(prob_ai * 100)))

        if risk < 30:
            level = "BAIXO"
        elif risk < 60:
            level = "MEDIO"
        elif risk < 80:
            level = "SUSPEITO"
        else:
            level = "ALTO"

        return {
            "risk": risk,
            "level": level,
            "ml_prediction": pred,
            "reasons": explain_features(features),
            "features": features,
        }

    except Exception as e:
        return {
            "error": "Erro ao analisar imagem",
            "detail": str(e),
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        return predict_video_with_hybrid_strategy(temp_path)

    except Exception as e:
        return {
            "error": "Erro ao analisar video",
            "detail": str(e),
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
