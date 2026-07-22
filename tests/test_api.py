import pytest
from fastapi.testclient import TestClient

from forensics.config import ARTIFACTS_DIR
from forensics.serving.api import app

client = TestClient(app)

ARTIFACTS_READY = (ARTIFACTS_DIR / "blender" / "calibrator.pkl").exists() and any(
    (ARTIFACTS_DIR / "encoder_folds").glob("fold*")
)


def test_health_reports_service_state():
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert body["encoder_folds_loaded"] > 0
        assert body["blender_folds_loaded"] > 0


@pytest.mark.skipif(not ARTIFACTS_READY, reason="trained model artifacts not present yet")
def test_predict_returns_valid_probability():
    resp = client.post("/predict", json={"text": "This is a short test sentence for the API."})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability_machine_generated"] <= 1.0
    assert body["label"] in ("human-written", "machine-generated")
    assert "encoder_prob" in body["detectors"]


def test_predict_rejects_empty_text():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422
