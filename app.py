from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from feature_extractor import compute_features
from risk_scoring import compute_risk_score

app = FastAPI(title="AI Video Detector")

# -------------------------
# CORS (frontend)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Endpoint principal
# -------------------------
@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Query("normal", enum=["normal", "strict"])
):
    try:
        # Guardar vídeo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        # Extrair features
        features = compute_features(video_path)

        # Calcular risco
        risk, reasons = compute_risk_score(features, mode=mode)

        # Classificação humana
        if risk < 30:
            level = "BAIXO"
        elif risk < 60:
            level = "MÉDIO"
        else:
            level = "ALTO"

        return {
            "risk": risk,
            "level": level,
            "reasons": reasons,
            "features": features,
        }

    except Exception as e:
        return {
            "error": "Erro ao analisar vídeo",
            "detail": str(e),
        }

    finally:
        if "video_path" in locals() and os.path.exists(video_path):
            os.remove(video_path)

