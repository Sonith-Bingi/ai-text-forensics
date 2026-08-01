from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)


class DetectorBreakdown(BaseModel):
    encoder_prob: float
    stat_loglik: float
    stat_logrank: float
    stat_lrr: float
    stat_curvature: float
    stylometric_burstiness: float
    stylometric_rep3_rate: float


class PredictResponse(BaseModel):
    probability_machine_generated: float
    label: str
    word_count: int
    reliability: str
    reliability_measured_accuracy: float
    detectors: DetectorBreakdown


class HealthResponse(BaseModel):
    status: str
    encoder_folds_loaded: int
    blender_folds_loaded: int
