from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from feature_extractor import compute_features
from risk_scoring import compute_risk_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Query("normal", enum=["normal", "strict"]),
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        features = compute_features(video_path)
        risk, level, reasons = compute_risk_score(features, mode)

        os.remove(video_path)

        return {
            "risk": risk,
            "level": level,
            "reasons": reasons,
            "features": features,
        }

    except Exception as e:
        return {"error": "Erro ao analisar vídeo", "detail": str(e)}

