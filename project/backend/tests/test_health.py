from app.main import app
from tests.asgi_client import ASGITestClient

client = ASGITestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "backend"}


def test_independent_flow_status_endpoints() -> None:
    expected_statuses = {
        "questionnaire": "ready",
        "stt": "ready",
        "tts": "ready",
        "prediction": "scaffolded",
    }
    for flow, expected_status in expected_statuses.items():
        response = client.get(f"/api/flows/{flow}/status")

        assert response.status_code == 200
        body = response.json()
        assert body["flow"] == flow
        assert body["status"] == expected_status


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
    assert body["audio_execution"]["tts_length_scale"] == 1.2
