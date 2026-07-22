from __future__ import annotations

from fastapi import FastAPI, HTTPException

from forensics.inference import get_predictor
from forensics.serving.schemas import HealthResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="AI Text Forensics",
    description="Machine-generated text detection: LoRA-tuned encoder + zero-shot "
    "statistical detectors + stylometric features, blended and calibrated.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifacts not found: {e}") from e
    return HealthResponse(
        status="ok",
        encoder_folds_loaded=len(predictor.encoder_models),
        blender_folds_loaded=len(predictor.blender_models),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor = get_predictor()
    result = predictor.predict(request.text)
    return PredictResponse(**result)
