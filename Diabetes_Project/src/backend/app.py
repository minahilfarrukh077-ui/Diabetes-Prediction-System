from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.backend.predictor import (
    explain_prediction,
    get_explain_summary,
    get_metadata,
    load_model_bundle,
    predict_diabetes,
)
from src.backend.schemas import (
    ExplainResponse,
    ExplainSummaryResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0.0",
    description="Python backend API for diabetes prediction using a trained sklearn pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model_bundle()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="model_not_ready", model_loaded=False)


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    try:
        return get_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return predict_diabetes(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for production failures
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.") from exc


@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictionRequest) -> ExplainResponse:
    try:
        return explain_prediction(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for production failures
        logger.exception("Explanation failed.")
        raise HTTPException(status_code=500, detail="Explanation failed unexpectedly.") from exc


@app.get("/explain/summary", response_model=ExplainSummaryResponse)
def explain_summary() -> ExplainSummaryResponse:
    try:
        return get_explain_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
