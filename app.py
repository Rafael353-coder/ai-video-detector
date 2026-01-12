from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from feature_extractor import compute_features
from risk_scoring import compute_risk_score

import tempfile
import shutil

app = FastAPI(title="AI Video Risk Detector")

# Servir frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Query("normal")
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            video_path = tmp.name

        features = compute_features(video_path)
        risk, level, reasons = compute_risk_score(features, mode)

        return {
            "risk": risk,
            "level": level,
            "reasons": reasons,
            "features": features
        }

    except Exception as e:
        return {
            "error": "Erro ao analisar vídeo",
            "detail": str(e)
        }

