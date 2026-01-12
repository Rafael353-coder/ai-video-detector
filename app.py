from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os

from feature_extractor import compute_features
from risk_scoring import compute_risk_score

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(
    title="AI Video Detector",
    description="API para deteção de vídeos gerados por IA",
    version="1.0"
)

# --------------------------------------------------
# CORS (frontend local / deploy)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # MVP / demo
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# ANALYZE ENDPOINT
# --------------------------------------------------
@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = "normal"   # normal | strict
):
    """
    Recebe um vídeo e devolve:
    - risk (0–100)
    - level (BAIXO | MÉDIO | ALTO)
    - reasons (explicáveis)
    - features (debug)
    """

    # -------------------------
    # Guardar vídeo temporário
    # -------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        video_path = tmp.name

    try:
        # -------------------------
        # Extrair features
        # -------------------------
        features = compute_features(video_path)

        # -------------------------
        # Calcular risco
        # -------------------------
        risk, level, reasons = compute_risk_score(features, mode)

        response = {
            "risk": risk,
            "level": level,
            "reasons": reasons,
            "features": features
        }

    except Exception as e:
        response = {
            "error": "Erro ao analisar vídeo",
            "detail": str(e)
        }

    finally:
        # -------------------------
        # Limpeza
        # -------------------------
        if os.path.exists(video_path):
            os.remove(video_path)

    return response

