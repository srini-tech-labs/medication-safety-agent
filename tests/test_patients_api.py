from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.databricks_client import DatabricksClientError

client = TestClient(app)

PATIENT_PAYLOAD = {
    "patient_id": "P003",
    "age": 72,
    "medications": ["Spironolactone", "Ibuprofen"],
    "conditions": [],
    "labs": {"egfr": 48, "potassium": 4.6},
    "allergies": [],
}


@patch("app.api.patients.analyze_patient")
def test_analyze_returns_structured_report(mock_analyze):
    mock_analyze.return_value = {
        "patient_id": "P003",
        "overall_risk": "MODERATE",
        "normalized_medications": ["Spironolactone", "Ibuprofen"],
        "drug_drug_findings": [],
        "drug_lab_findings": [{"rule": "K + Spironolactone"}],
        "ai_explanation": "Monitor potassium levels.",
        "explanation_provider": "Databricks",
        "explanation_model": "meta-llama-3.3-70b-instruct-121024",
    }

    response = client.post("/api/patients/analyze", json=PATIENT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["overall_risk"] == "MODERATE"
    assert body["explanation_provider"] == "Databricks"
    assert body["explanation_model"] == "meta-llama-3.3-70b-instruct-121024"
    mock_analyze.assert_called_once_with(PATIENT_PAYLOAD)


@patch("app.api.patients.analyze_patient")
def test_analyze_maps_databricks_error_to_502(mock_analyze):
    mock_analyze.side_effect = DatabricksClientError("endpoint not found")

    response = client.post("/api/patients/analyze", json=PATIENT_PAYLOAD)

    assert response.status_code == 502
    assert "endpoint not found" in response.json()["detail"]


def test_analyze_rejects_missing_required_fields():
    response = client.post("/api/patients/analyze", json={"patient_id": "P001"})

    assert response.status_code == 422


def test_home_page_renders_seed_patients():
    response = client.get("/")

    assert response.status_code == 200
    assert "P001" in response.text
