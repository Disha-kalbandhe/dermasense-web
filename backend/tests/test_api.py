import io
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def client():
    try:
        from fastapi.testclient import TestClient
        from app.main import app
    except Exception as exc:
        pytest.skip(f"Backend dependencies are not importable in this environment: {exc}")
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_without_checkpoint_reports_real_blocker(client):
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/predict",
        files={"image": ("sample.png", buffer.getvalue(), "image/png")},
        data={"age": "35", "gender": "Female", "skin_tone": "IV", "body_location": "Face", "duration": "1-4 weeks", "symptoms": "[]"},
    )
    checkpoint = Path(__file__).resolve().parents[2] / "ml" / "checkpoints" / "best_model.pth"
    if checkpoint.exists():
        assert response.status_code == 200
        assert response.json()["explainability"]["available"] is True
    else:
        assert response.status_code == 500
        assert "checkpoint" in response.json()["detail"].lower()
