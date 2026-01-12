from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from feature_extractor import compute_features
from risk_scoring import compute_risk_score

app = FastAPI(title="AI Video Detector")

# Permitir frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "API online"}


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    mode: str = Query("normal", enum=["normal", "strict"])
):
    temp_path = None

    try:
        # Guardar vídeo temporário
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        # Extrair features
        features = compute_features(temp_path)

        # Calcular risco
        risk, level, reasons = compute_risk_score(
            features,
            mode=mode
        )

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

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

