"""
FastAPI application — Phage Therapy Prediction API.

Endpoints:
  POST /api/predict        Upload a FASTA file and get ranked phage predictions.
  GET  /api/models         List loaded models.
  GET  /api/health         Health check.
  GET  /api/phages         List all phages in the library.
  GET  /api/results        Training results (metrics, CV folds).
  GET  /api/training-info  Dataset statistics.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --------------- project root on sys.path ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------------------------------------

from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger
from backend.services.file_handler import (
    validate_fasta_file,
    save_uploaded_file,
)
from backend.services.prediction_service import PredictionService

logger = setup_logger(__name__, level=logging.INFO)

# --------------- singleton service ----------------------------------------
prediction_service = PredictionService()


# --------------- lifespan (startup / shutdown) ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("Starting up — loading models…")
    prediction_service.load()
    logger.info("Models loaded — API ready")
    yield
    logger.info("Shutting down")


# --------------- app creation ---------------------------------------------
app = FastAPI(
    title="Phage Therapy Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------- routes ---------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": prediction_service.is_loaded,
        "available_models": prediction_service.available_models,
    }


@app.get("/api/models")
async def list_models():
    return {
        "models": prediction_service.available_models,
    }


@app.get("/api/phages")
async def list_phages():
    """Return all phages known in the library (with morphology)."""
    meta_path = Path(config.paths["phage_library"]) / "phage_metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "Phage metadata not found")
    with open(meta_path) as f:
        metadata = json.load(f)
    return {"phages": metadata, "count": len(metadata)}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=500),
    view: str = Query("phage", pattern="^(phage|interaction)$"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Upload a clinical-isolate FASTA and receive ranked phage predictions
    from ALL models.

    Query params:
      - **top_k**: Number of top phages to return per model
      - **view**: `phage` (all phage × concentration combos) or
                  `interaction` (unique phages, best interaction)
      - **threshold**: Feasibility CI-score threshold
    """
    # 1. Read & validate
    content = await file.read()
    error = validate_fasta_file(file.filename, content)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # 2. Save to temp
    fasta_path = await save_uploaded_file(file.filename, content)

    try:
        # 3. Run pipeline (all models) — files persist in upload folder
        result = prediction_service.predict(
            fasta_path=fasta_path,
            top_k=top_k,
            view=view,
            threshold=threshold,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results")
async def get_results():
    """Return training results: model_comparison + per-model CV & test metrics."""
    results_dir = Path(config.paths["results"])
    if not results_dir.exists():
        raise HTTPException(404, "Results directory not found")

    def read_csv(p: Path):
        if not p.exists():
            return []
        with open(p) as f:
            return list(csv.DictReader(f))

    # model_comparison.csv
    comparison = read_csv(results_dir / "model_comparison.csv")
    for row in comparison:
        for k, v in row.items():
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                pass

    # Per-model CV + test results
    per_model = {}
    for model_file in sorted(results_dir.glob("*_cv_results.csv")):
        name = model_file.stem.replace("_cv_results", "")
        cv = read_csv(model_file)
        test = read_csv(results_dir / f"{name}_test_results.csv")
        for rows in (cv, test):
            for row in rows:
                for k, v in row.items():
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        pass
        per_model[name] = {"cv": cv, "test": test}

    return {
        "comparison": comparison,
        "per_model": per_model,
    }


@app.get("/api/training-info")
async def get_training_info():
    """Return dataset statistics used for training."""
    interactions_path = Path(config.paths["interactions"])
    if not interactions_path.exists():
        raise HTTPException(404, "Interactions data not found")

    df = pd.read_csv(interactions_path)
    train = df[df["dataset"] == "train"]
    test = df[df["dataset"] == "test"]

    return {
        "total_samples": len(df),
        "train_samples": len(train),
        "test_samples": len(test),
        "positive_samples": int((df["class"] == 1).sum()),
        "negative_samples": int((df["class"] == 0).sum()),
        "unique_phages": int(df["phage"].nunique()),
        "unique_hosts": int(df["host"].nunique()),
        "morphologies": df["morphology"].value_counts().to_dict(),
        "concentrations": sorted(df["concentration"].unique().tolist()),
        "cv_folds": config.training.get("cv_folds", 10),
        "random_seed": config.training.get("random_seed", 42),
    }


# --------------- entry point ----------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = config.api.get("host", "0.0.0.0")
    port = config.api.get("port", 8000)
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
