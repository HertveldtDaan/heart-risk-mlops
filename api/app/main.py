"""Heart Risk API: FastAPI wrapper around the model trained in Azure ML.

Endpoints
  POST /predict             risk estimate + top 5 SHAP factors, stored in the database
  GET  /predictions         history of stored predictions (newest first)
  GET  /predictions/{id}    one stored prediction
  GET  /predictions/stats   simple monitoring: volume and risk distribution per model version
  GET  /model/info          threshold, test metrics, fairness audit, global importance
  GET  /version             lineage: model version, training commit, API commit, image
  GET  /health  /ready      Kubernetes liveness and readiness probes
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select

from app.db import Database, Prediction, database_url
from app.model_service import DISCLAIMER, ModelService
from app.schemas import ModelRef, PatientInput, PredictionRecord, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelService(os.getenv("MODEL_DIR", "model"))
    print(f"Model {app.state.model.name} v{app.state.model.version} loaded "
          f"(threshold {app.state.model.threshold}).")
    app.state.db = Database(database_url())
    app.state.db.init()
    print("Database ready.")
    yield
    app.state.db.engine.dispose()


app = FastAPI(
    title="Heart Risk API",
    description="Estimates the heart attack risk profile from a health questionnaire (CDC BRFSS 2022). "
                "Not a medical diagnosis.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["operations"])
def health():
    """Liveness: the process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["operations"])
def ready(response: Response):
    """Readiness: model loaded AND database reachable. Kubernetes only sends traffic when this is 200."""
    db_up = app.state.db.is_up()
    if not db_up:
        response.status_code = 503
    return {"model_loaded": app.state.model is not None, "database": "up" if db_up else "down"}


@app.get("/version", tags=["operations"])
def version():
    """Traceability: from a running API back to the image, the commit and the training run."""
    m = app.state.model
    return {
        "model_name": m.name,
        "model_version": m.version,
        "model_trained_at": m.release.get("created_at"),
        "training_git_commit": m.release.get("git_commit"),
        "azureml_run_id": m.release.get("azureml_run_id"),
        "mlflow_run_id": m.release.get("mlflow_run_id"),
        "data_sha256": m.release.get("data", {}).get("data_sha256"),
        "api_git_commit": os.getenv("GIT_COMMIT", "local"),
        "image": os.getenv("IMAGE_REF", "local"),
    }


@app.get("/model/info", tags=["model"])
def model_info():
    m = app.state.model
    return {
        "model": {"name": m.name, "version": m.version},
        "threshold": m.threshold,
        "risk_levels": {"low": f"< {m.threshold}",
                        "elevated": f">= {m.threshold}",
                        "high": f">= {min(0.9, 3 * m.threshold):.2f}"},
        "validation_metrics": m.release.get("validation_metrics"),
        "test_metrics": m.evaluation.get("test_metrics"),
        "quality_gate": m.evaluation.get("quality_gate"),
        "fairness": m.evaluation.get("fairness"),
        "shap_global_importance": m.evaluation.get("shap_global_importance"),
        "features": m.release.get("features"),
        "excluded_columns": m.release.get("excluded_columns"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(patient: PatientInput):
    m = app.state.model
    answers = patient.model_dump()
    result = m.predict(answers)

    record = Prediction(
        created_at=datetime.now(timezone.utc),
        model_name=m.name,
        model_version=m.version,
        input=answers,
        **result,
    )
    with app.state.db.session() as session:
        session.add(record)
        session.commit()
        session.refresh(record)

    return PredictionResponse(
        id=record.id,
        created_at=record.created_at,
        threshold=m.threshold,
        model=ModelRef(name=m.name, version=m.version),
        disclaimer=DISCLAIMER,
        **result,
    )


@app.get("/predictions", response_model=list[PredictionRecord], tags=["predictions"])
def list_predictions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    with app.state.db.session() as session:
        rows = session.scalars(
            select(Prediction).order_by(Prediction.id.desc()).offset(offset).limit(limit)
        ).all()
    return rows


@app.get("/predictions/stats", tags=["predictions"])
def prediction_stats():
    """Basic production monitoring: if the risk distribution suddenly shifts, the input data may have drifted."""
    with app.state.db.session() as session:
        rows = session.execute(
            select(
                Prediction.model_version,
                func.count(Prediction.id),
                func.avg(Prediction.risk_probability),
                func.min(Prediction.created_at),
                func.max(Prediction.created_at),
            ).group_by(Prediction.model_version)
        ).all()
        levels = session.execute(
            select(Prediction.model_version, Prediction.risk_level, func.count(Prediction.id))
            .group_by(Prediction.model_version, Prediction.risk_level)
        ).all()
    stats = {
        v: {"count": n, "avg_risk_probability": round(float(avg or 0), 4),
            "first": first, "last": last, "risk_levels": {}}
        for v, n, avg, first, last in rows
    }
    for v, level, n in levels:
        stats[v]["risk_levels"][level] = n
    return stats


@app.get("/predictions/{prediction_id}", response_model=PredictionRecord, tags=["predictions"])
def get_prediction(prediction_id: int):
    with app.state.db.session() as session:
        row = session.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return row
