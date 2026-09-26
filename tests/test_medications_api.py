from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.medication_management import MedicationManagementError

client = TestClient(app)


@patch("app.api.medications.medication_management.health")
def test_health_passthrough(mock_health):
    mock_health.return_value = {
        "status": "OK",
        "rxnorm": {"version": "08-Sep-2026", "apiVersion": "3.1.355"},
        "databricks_sql": [{"sql_status": "OK"}],
        "service": "medication-management (local)",
    }

    response = client.get("/api/medications/health")

    assert response.status_code == 200
    assert response.json()["status"] == "OK"
    mock_health.assert_called_once_with()


@patch("app.api.medications.medication_management.list_patient_medications")
def test_list_patient_medications_empty(mock_list):
    mock_list.return_value = {"patient_id": "P001", "medications": [], "count": 0}

    response = client.get("/api/medications/patients/P001")

    assert response.status_code == 200
    assert response.json() == {"patient_id": "P001", "medications": [], "count": 0}
    mock_list.assert_called_once_with("P001")


@patch("app.api.medications.medication_management.search_medications")
def test_search_medications_happy_path(mock_search):
    mock_search.return_value = {
        "query": "lisi",
        "matches": [{"rxcui": "372614", "name": "lisinopril Oral Tablet", "rank": 3, "score": 7.95, "search_scope": "PRESCRIBABLE_RXNORM"}],
        "count": 1,
        "cache": "miss",
        "source": "NLM RxNorm / Prescribable RxNorm",
    }

    response = client.get("/api/medications/search", params={"q": "lisi", "limit": 10})

    assert response.status_code == 200
    assert response.json()["matches"][0]["rxcui"] == "372614"
    mock_search.assert_called_once_with("lisi", 10)


@patch("app.api.medications.medication_management.search_medications")
def test_search_medications_rejects_short_query(mock_search):
    response = client.get("/api/medications/search", params={"q": "li"})

    assert response.status_code == 422
    mock_search.assert_not_called()


@patch("app.api.medications.medication_management.get_medication_details")
def test_get_medication_details_maps_error_to_502(mock_details):
    mock_details.side_effect = MedicationManagementError(
        "get_medication_details failed: Databricks SQL failed: PERMISSION_DENIED"
    )

    response = client.get("/api/medications/rxcui/372614")

    assert response.status_code == 502
    assert "PERMISSION_DENIED" in response.json()["detail"]


@patch("app.api.medications.medication_management.pre_add_check")
def test_pre_add_check_sends_none_labs_when_missing(mock_check):
    mock_check.return_value = {
        "patient_id": "P001",
        "drug_drug_findings": [],
        "drug_lab_findings": [],
        "total_findings": 0,
        "safety_review_required": False,
        "message": "No supported safety concern was identified.",
        "coverage_note": "Pre-add screening uses the prototype Databricks rule tables.",
    }

    response = client.post("/api/medications/pre-add-check", json={"patient_id": "P001", "rxcui": "372614"})

    assert response.status_code == 200
    mock_check.assert_called_once_with("P001", "372614", None)


@patch("app.api.medications.medication_management.pre_add_check")
def test_pre_add_check_forwards_labs_when_present(mock_check):
    mock_check.return_value = {"total_findings": 0, "safety_review_required": False}

    client.post(
        "/api/medications/pre-add-check",
        json={"patient_id": "P001", "rxcui": "372614", "labs": {"egfr": 55, "potassium": 5.1}},
    )

    mock_check.assert_called_once_with("P001", "372614", {"egfr": 55, "potassium": 5.1})


@patch("app.api.medications.medication_management.add_patient_medication")
def test_add_patient_medication_happy_path(mock_add):
    mock_add.return_value = {
        "added": True,
        "duplicate": False,
        "medication_record_id": "d6c0b22f-417e-4f4c-8038-08a7f8b7c60f",
        "patient_id": "P001",
        "medication_type": "UNKNOWN",
        "medication_status": "PROPOSED",
        "safety": {"total_findings": 1, "safety_review_required": True},
        "message": "Medication saved as PROPOSED. Safety findings were identified; "
        "review them with the prescribing clinician or pharmacist before starting the medication.",
    }

    response = client.post(
        "/api/medications/add",
        json={
            "patient_id": "P001",
            "rxcui": "372614",
            "medication_status": "PROPOSED",
            "medication_type": "UNKNOWN",
            "labs": {"egfr": 55, "potassium": 5.1},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["added"] is True
    assert body["duplicate"] is False
    mock_add.assert_called_once_with("P001", "372614", "PROPOSED", "UNKNOWN", {"egfr": 55, "potassium": 5.1})


@patch("app.api.medications.medication_management.add_patient_medication")
def test_add_patient_medication_duplicate(mock_add):
    mock_add.return_value = {
        "added": False,
        "duplicate": True,
        "patient_id": "P001",
        "selected_rxcui": "372614",
        "existing_record": {
            "medication_record_id": "d6c0b22f-417e-4f4c-8038-08a7f8b7c60f",
            "selected_name": "lisinopril Oral Tablet",
            "medication_status": "PROPOSED",
        },
        "message": "This medication is already active on the patient list.",
    }

    response = client.post(
        "/api/medications/add",
        json={
            "patient_id": "P001",
            "rxcui": "372614",
            "medication_status": "PROPOSED",
            "medication_type": "UNKNOWN",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["duplicate"] is True
    assert body["existing_record"]["medication_status"] == "PROPOSED"


@patch("app.api.medications.medication_management.remove_patient_medication")
def test_discontinue_medication_happy_path(mock_remove):
    mock_remove.return_value = {
        "removed": True,
        "patient_id": "P001",
        "medication_record_id": "d6c0b22f-417e-4f4c-8038-08a7f8b7c60f",
        "message": "Medication marked DISCONTINUED and retained for history.",
    }

    response = client.post(
        "/api/medications/discontinue",
        json={"patient_id": "P001", "medication_record_id": "d6c0b22f-417e-4f4c-8038-08a7f8b7c60f"},
    )

    assert response.status_code == 200
    assert response.json()["removed"] is True
    mock_remove.assert_called_once_with("P001", "d6c0b22f-417e-4f4c-8038-08a7f8b7c60f")
