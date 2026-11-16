#!/usr/bin/env python
# predict.py — FastAPI service with both a JSON API and a minimal HTML UI
#
# Endpoints:
#   GET  /              → HTML form (crop dropdown, model dropdown, feature inputs)
#   GET  /health        → {"status": "ok", "models_loaded": N}
#   GET  /crops         → {"crops": [...], "model_types": [...], "models": {...}}
#   GET  /schema        → the raw feature_schema.json
#   GET  /docs          → Swagger UI (auto)
#   GET  /redoc         → ReDoc (auto)
#   POST /predict       → {"crop": str, "model": str, "features": {...}}
#
# Run:
#   uvicorn predict:app --host 0.0.0.0 --port 9696
#
# Config sources:
#   models/feature_schema.json  → which crops + model families exist, feature list
#   config.yaml `training:`     → model_types, default_model_type, models_dir
#   config.yaml `serving:`      → port, title

import json
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
# Paths and config
# ══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE  = PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    if not CONFIG_FILE.is_file():
        return {}
    try:
        with CONFIG_FILE.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


CFG = _load_config()


def _cfg(*keys, default):
    node = CFG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default


MODELS_DIR          = PROJECT_ROOT / _cfg("training", "models_dir", default="models")
SCHEMA_FILE         = MODELS_DIR / "feature_schema.json"
DEFAULT_PORT        = int(os.getenv("PORT", _cfg("serving", "port", default=9696)))
DEFAULT_MODEL_TYPE  = _cfg("training", "default_model_type", default="dt")
ALLOWED_MODEL_TYPES = _cfg("training", "model_types", default=["dt", "xgb"])
APP_TITLE           = _cfg("serving", "title", default="FTL Syria — Crop Yield Prediction")


# ══════════════════════════════════════════════════════════════════
# Load schema
# ══════════════════════════════════════════════════════════════════
if not SCHEMA_FILE.is_file():
    raise RuntimeError(
        f"Missing {SCHEMA_FILE}. Train models first: python train.py"
    )

with SCHEMA_FILE.open(encoding="utf-8") as f:
    SCHEMA = json.load(f)

FEATURE_COLS: list[str]                    = SCHEMA["feature_cols"]
TARGET: str                                = SCHEMA["target"]
AVAILABLE_CROPS: list[str]                 = SCHEMA["crops"]
AVAILABLE_MODELS: dict[str, dict[str, str]] = SCHEMA["models"]


# ══════════════════════════════════════════════════════════════════
# Model cache
# ══════════════════════════════════════════════════════════════════
class ModelCache:
    def __init__(self):
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, crop: str, family: str):
        key = (crop, family)
        if key in self._cache:
            return self._cache[key]

        filename = AVAILABLE_MODELS.get(crop, {}).get(family)
        if filename is None:
            raise FileNotFoundError(
                f"No trained '{family}' model for crop '{crop}'. "
                f"Available: {list(AVAILABLE_MODELS.get(crop, {}).keys())}"
            )

        path = MODELS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}. "
                f"Re-run: python train.py --crop {crop} --model {family}"
            )

        model = joblib.load(path)
        self._cache[key] = model
        return model

    def loaded_count(self) -> int:
        return len(self._cache)

    def preload_all(self):
        for crop, families in AVAILABLE_MODELS.items():
            for family in families:
                try:
                    self.get(crop, family)
                except Exception as e:
                    print(f"[warn] preload {crop}/{family} failed: {e}", flush=True)


CACHE = ModelCache()
FEATURE_DEFAULTS: dict[str, float] = {c: 0.0 for c in FEATURE_COLS}


# ══════════════════════════════════════════════════════════════════
# Request / response models
# ══════════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    crop: str = Field(..., description=f"One of: {AVAILABLE_CROPS}")
    model: Optional[str] = Field(
        default=None,
        description=f"Model family. One of: {ALLOWED_MODEL_TYPES}. "
                    f"Default: {DEFAULT_MODEL_TYPE}",
    )
    features: dict[str, float] = Field(
        ...,
        description=f"Required keys: {FEATURE_COLS}",
    )


class PredictResponse(BaseModel):
    crop: str
    model: str
    predicted_yield_hg_ha: float
    input_features: dict[str, float]


# ══════════════════════════════════════════════════════════════════
# Shared prediction logic (used by both API and UI)
# ══════════════════════════════════════════════════════════════════
def _predict(crop: str, family: str, features: dict[str, float]) -> float:
    """Validate, load, and predict. Raises HTTPException on any failure."""
    if crop not in AVAILABLE_CROPS:
        raise HTTPException(404,
            f"Unknown crop '{crop}'. Available: {AVAILABLE_CROPS}")
    if family not in ALLOWED_MODEL_TYPES:
        raise HTTPException(400,
            f"Unknown model '{family}'. Allowed: {ALLOWED_MODEL_TYPES}")

    provided = set(features.keys())
    required = set(FEATURE_COLS)
    missing = required - provided
    extra   = provided - required
    if missing:
        raise HTTPException(422, f"Missing feature(s): {sorted(missing)}")
    if extra:
        raise HTTPException(422, f"Unexpected feature(s): {sorted(extra)}")

    for k, v in features.items():
        if not isinstance(v, (int, float)):
            raise HTTPException(422,
                f"Feature '{k}' must be numeric, got {type(v).__name__}")

    try:
        model = CACHE.get(crop, family)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    X = pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    try:
        return float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title=APP_TITLE,
    description=(
        "Per-crop yield prediction from climate + anomaly features. "
        "Interactive UI at `/`. Swagger docs at `/docs`. "
        "JSON API at `POST /predict`."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _startup():
    global FEATURE_DEFAULTS
    print(f"[startup] schema: {SCHEMA_FILE}", flush=True)
    print(f"[startup] crops:  {AVAILABLE_CROPS}", flush=True)
    print(f"[startup] models: {AVAILABLE_MODELS}", flush=True)

    # Compute per-feature means from the training split for the UI defaults
    train_path = PROJECT_ROOT / "data" / "interim" / "train.csv"
    if train_path.is_file():
        train = pd.read_csv(train_path)
        avail = [c for c in FEATURE_COLS if c in train.columns]
        FEATURE_DEFAULTS = {
            c: float(round(train[c].mean(), 2)) for c in avail
        }
        print(f"[startup] feature defaults (from train.csv): {FEATURE_DEFAULTS}",
              flush=True)

    CACHE.preload_all()
    print(f"[startup] preloaded {CACHE.loaded_count()} models", flush=True)


# ══════════════════════════════════════════════════════════════════
# JSON API
# ══════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": CACHE.loaded_count()}


@app.get("/crops")
def crops():
    return {
        "crops": AVAILABLE_CROPS,
        "model_types": ALLOWED_MODEL_TYPES,
        "default_model_type": DEFAULT_MODEL_TYPE,
        "models": AVAILABLE_MODELS,
    }


@app.get("/schema")
def schema():
    return SCHEMA


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    family = req.model or DEFAULT_MODEL_TYPE
    y = _predict(req.crop, family, req.features)
    return PredictResponse(
        crop=req.crop,
        model=family,
        predicted_yield_hg_ha=y,
        input_features=req.features,
    )


# ══════════════════════════════════════════════════════════════════
# HTML UI at /
# ══════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def index():
    crop_options = "".join(
        f'<option value="{c}">{c}</option>' for c in AVAILABLE_CROPS
    )
    model_options = "".join(
        f'<option value="{m}"'
        f'{" selected" if m == DEFAULT_MODEL_TYPE else ""}>{m}</option>'
        for m in ALLOWED_MODEL_TYPES
    )

    feature_inputs = "".join(
        f'''
        <div class="field">
            <label for="{f}">{f}</label>
            <input type="number" step="any" id="{f}" name="{f}"
                   value="{FEATURE_DEFAULTS.get(f, 0)}" required>
        </div>
        '''
        for f in FEATURE_COLS
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{APP_TITLE}</title>
<style>
    * {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, system-ui, sans-serif;
        max-width: 760px;
        margin: 40px auto;
        padding: 0 20px;
        color: #1a1a1a;
        background: #fafbfc;
    }}
    h1 {{ margin: 0 0 4px; font-size: 1.6em; }}
    .sub {{ color: #666; margin: 0 0 24px; }}
    .links {{ margin-bottom: 24px; font-size: 0.9em; }}
    .links a {{ color: #2563eb; text-decoration: none; margin-right: 16px; }}
    .links a:hover {{ text-decoration: underline; }}

    form {{
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 24px;
    }}
    h3 {{
        font-size: 1em;
        margin: 20px 0 8px;
        color: #333;
        border-bottom: 1px solid #eee;
        padding-bottom: 6px;
    }}
    .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0 16px;
    }}
    @media (max-width: 600px) {{
        .grid {{ grid-template-columns: 1fr; }}
    }}
    .field {{ margin: 10px 0; }}
    label {{
        display: block;
        font-weight: 600;
        font-size: 0.85em;
        color: #444;
        margin-bottom: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    select, input {{
        width: 100%;
        padding: 8px 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        font-size: 0.95em;
        font-family: inherit;
        background: white;
    }}
    select:focus, input:focus {{
        outline: none;
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
    }}
    button {{
        margin-top: 20px;
        padding: 10px 28px;
        background: #2563eb;
        color: white;
        border: 0;
        border-radius: 6px;
        font-size: 1em;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
    }}
    button:hover {{ background: #1d4ed8; }}
    button:disabled {{ background: #94a3b8; cursor: not-allowed; }}

    #result, #error {{
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        display: none;
    }}
    #result {{ background: #eff6ff; border: 1px solid #bfdbfe; }}
    #result .value {{
        font-size: 2em;
        font-weight: 700;
        color: #1d4ed8;
        margin: 0 0 6px;
        font-variant-numeric: tabular-nums;
    }}
    #result .meta {{ color: #475569; font-size: 0.9em; }}
    #error {{ background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c; }}
    #error pre {{ margin: 8px 0 0; white-space: pre-wrap; font-size: 0.85em; }}
</style>
</head>
<body>
    <h1>{APP_TITLE}</h1>
    <p class="sub">Enter climate and anomaly features to predict crop yield (hg/ha).</p>

    <div class="links">
        <a href="/docs">Swagger UI</a>
        <a href="/redoc">ReDoc</a>
        <a href="/crops">Available crops &amp; models</a>
        <a href="/health">Health</a>
    </div>

    <form id="form">
        <div class="grid">
            <div class="field">
                <label for="crop">crop</label>
                <select name="crop" id="crop">{crop_options}</select>
            </div>
            <div class="field">
                <label for="model">model</label>
                <select name="model" id="model">{model_options}</select>
            </div>
        </div>

        <h3>Features</h3>
        <div class="grid">{feature_inputs}</div>

        <button type="submit" id="submit">Predict</button>
    </form>

    <div id="result">
        <div class="value" id="result-value"></div>
        <div class="meta" id="result-meta"></div>
    </div>
    <div id="error">
        <strong>Error</strong>
        <pre id="error-detail"></pre>
    </div>

<script>
document.getElementById("form").addEventListener("submit", async (e) => {{
    e.preventDefault();

    const btn = document.getElementById("submit");
    btn.disabled = true;
    btn.textContent = "Predicting…";

    const resultEl = document.getElementById("result");
    const errorEl  = document.getElementById("error");
    resultEl.style.display = "none";
    errorEl.style.display  = "none";

    const fd = new FormData(e.target);
    const features = {{}};
    for (const [k, v] of fd.entries()) {{
        if (k === "crop" || k === "model") continue;
        features[k] = parseFloat(v);
    }}
    const payload = {{
        crop: fd.get("crop"),
        model: fd.get("model"),
        features: features,
    }};

    try {{
        const res = await fetch("/predict", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(payload),
        }});
        const data = await res.json();

        if (!res.ok) {{
            document.getElementById("error-detail").textContent =
                JSON.stringify(data, null, 2);
            errorEl.style.display = "block";
        }} else {{
            document.getElementById("result-value").textContent =
                data.predicted_yield_hg_ha.toLocaleString(undefined, {{
                    maximumFractionDigits: 1,
                }}) + " hg/ha";
            document.getElementById("result-meta").textContent =
                `Crop: ${{data.crop}}  ·  Model: ${{data.model}}`;
            resultEl.style.display = "block";
        }}
    }} catch (err) {{
        document.getElementById("error-detail").textContent =
            "Network error: " + err;
        errorEl.style.display = "block";
    }} finally {{
        btn.disabled = false;
        btn.textContent = "Predict";
    }}
}});
</script>
</body>
</html>
    """
    return HTMLResponse(content=html)


# ══════════════════════════════════════════════════════════════════
# Local dev
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="0.0.0.0", port=DEFAULT_PORT, reload=False)