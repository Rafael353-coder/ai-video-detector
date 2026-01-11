from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os

from feature_extractor import compute_features
from risk_scoring import compute_risk_score

app = FastAPI(title="AI Video Risk Detector")

# =========================
# CORS (para frontend local)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Health check
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "AI Video Risk Detector running"}

# =========================
# Analyze endpoint
# =========================
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # Guardar vídeo temporariamente
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, file.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 1️⃣ Extrair features
        features = compute_features(video_path)

        # 2️⃣ Calcular risco
        risk, reasons = compute_risk_score(features)

        # 3️⃣ Classificação textual
        if risk >= 70:
            level = "ALTO"
        elif risk >= 40:
            level = "MÉDIO"
        else:
            level = "BAIXO"

        return {
            "risk": risk,
            "level": level,
            "reasons": reasons,
            "features": features
        }

    finally:
        shutil.rmtree(tmp_dir)

from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), 
name="frontend")

