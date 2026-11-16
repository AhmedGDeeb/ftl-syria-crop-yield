import os, json, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Crop Yield Prediction API")

# Load schema and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "models", "feature_schema.json")
with open(SCHEMA_PATH) as f:
    schema = json.load(f)

FEATURE_COLS = schema["feature_cols"]
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Cache for loaded models (load on first request to avoid startup timeout)
_model_cache: Dict[str, object] = {}

def get_model(crop: str, family: str):
    key = f"{crop}_{family}"
    if key not in _model_cache:
        filename = schema["models"][crop][family]
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise HTTPException(404, f"Model not found: {filename}")
        _model_cache[key] = joblib.load(path)
    return _model_cache[key]

class PredictRequest(BaseModel):
    crop: str
    family: str = "xgb"  # default
    features: List[float]

class PredictResponse(BaseModel):
    crop: str
    family: str
    prediction: float

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.crop not in schema["crops"]:
        raise HTTPException(400, f"Unknown crop: {req.crop}")
    if req.family not in schema["families"]:
        raise HTTPException(400, f"Unknown family: {req.family}")
    if len(req.features) != len(FEATURE_COLS):
        raise HTTPException(400, f"Expected {len(FEATURE_COLS)} features, got {len(req.features)}")

    model = get_model(req.crop, req.family)
    # XGBoost / sklearn predict expects 2D array
    import numpy as np
    X = np.array(req.features).reshape(1, -1)
    pred = model.predict(X)[0]
    return PredictResponse(crop=req.crop, family=req.family, prediction=float(pred))