from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "backend"}


def test_independent_flow_status_endpoints() -> None:
    for flow in ("questionnaire", "stt", "tts", "prediction"):
        response = client.get(f"/api/flows/{flow}/status")

        assert response.status_code == 200
        body = response.json()
        assert body["flow"] == flow
        assert body["status"] == "scaffolded"


def test_questionnaire_schema_and_prediction_models() -> None:
    questionnaire_response = client.get("/api/flows/questionnaire/schema")
    prediction_response = client.get("/api/flows/prediction/models")

    assert questionnaire_response.status_code == 200
    assert prediction_response.status_code == 200
    assert isinstance(questionnaire_response.json().get("questions"), list)
    assert isinstance(prediction_response.json().get("models"), list)


def test_runtime_config() -> None:
    response = client.get("/api/runtime/config")

    assert response.status_code == 200
    body = response.json()
    assert body["runtime_mode"] == "raspi-local"
    assert "audio_execution" in body
