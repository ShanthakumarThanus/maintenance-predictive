from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Maintenance Prédictive API")

# Chargement du modèle (fictif pour l'instant)
# model = joblib.load("../models/model.pkl")

# Schéma des données d'entrée
class MachineData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    rpm: float
    pressure_level: float

@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API Maintenance Prédictive",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

# Endpoint de santé
@app.get("/health")
def health():
    return {"status": "ok", "model": "chargé"}

# Endpoint de prédiction
@app.post("/predict")
def predict(data: MachineData):
    try:
        # Modèle fictif pour l'instant → retourne toujours 0
        # Quand le .pkl sera disponible, remplacer par :
        # features = [[data.vibration_rms, data.temperature_motor, 
        #              data.rpm, data.pressure_level]]
        # prediction = model.predict(features)[0]
        # probability = model.predict_proba(features)[0][1]

        prediction = 0          # fictif
        probability = 0.12      # fictif

        return {
            "failure_within_24h": int(prediction),
            "probability": round(probability, 4),
            "message": "Aucune panne détectée" if prediction == 0 else "Panne probable !"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))