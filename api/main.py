import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Maintenance Prédictive API")

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]

class MachineData(BaseModel):
    vibration_rms: float = Field(ge=0)
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float = Field(ge=0)
    rpm: float = Field(ge=0)
    hours_since_maintenance: float = Field(ge=0)
    ambient_temp: float

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API Maintenance Prédictive",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    try:
        metrics_path = Path("../model/metrics.json")
        fi_path = Path("../model/feature_importance.json")

        with open(metrics_path) as f:
            metrics = json.load(f)

        with open(fi_path) as f:
            feature_importance = json.load(f)

        all_models = metrics.get("models_sans_smote", []) + metrics.get("models_avec_smote", [])

        return {
            "metrics": {
                "models": all_models,
                "selected_model": metrics.get("selected_model", "n/a")
            },
            "feature_importance": feature_importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(data: MachineData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé : model/model.pkl est introuvable")

    try:
        features = pd.DataFrame([data.model_dump()], columns=FEATURES)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        return {
            "failure_within_24h": prediction,
            "probability": round(probability, 4),
            "message": "Panne probable !" if prediction == 1 else "Aucune panne détectée",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
