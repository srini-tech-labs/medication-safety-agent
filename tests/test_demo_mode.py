from unittest.mock import patch

from app.config import settings
from app.services import demo_store
from app.services.databricks_client import analyze_patient, check_pre_add
from app.services.sql_client import execute_sql


def _demo_mode():
    return patch.object(settings, "demo_mode", True)


@_demo_mode()
def test_analyze_patient_uses_canned_fixture_for_seed_patient():
    with patch("app.services.databricks_client.WorkspaceClient") as mock_ws:
        result = analyze_patient({"patient_id": "P001", "age": 68, "medications": ["Lisinopril"]})

    mock_ws.assert_not_called()
    assert result["patient_id"] == "P001"
    assert result["overall_risk"] == "HIGH"
    assert "[DEMO MODE]" in result["ai_explanation"]
    assert result["explanation_provider"] == "Demo (offline, canned fixture)"


@_demo_mode()
def test_analyze_patient_falls_back_for_unknown_patient():
    with patch("app.services.databricks_client.WorkspaceClient") as mock_ws:
        result = analyze_patient({"patient_id": "P999", "age": 40, "medications": []})

    mock_ws.assert_not_called()
    assert result["overall_risk"] == "LOW"
    assert "no canned analysis" in result["ai_explanation"]


@_demo_mode()
def test_check_pre_add_uses_canned_rule_table():
    context = {
        "candidate_name": "spironolactone",
        "candidate_ingredients": [{"rxcui": "9997", "name": "spironolactone"}],
        "current_medications": [
            {"selected_name": "lisinopril", "ingredients": [{"rxcui": "29046", "name": "lisinopril"}]}
        ],
    }
    with patch("app.services.databricks_client.WorkspaceClient") as mock_ws:
        result = check_pre_add(candidate_rxcui="9997", context=context, labs={"egfr": 60, "potassium": 5.5})

    mock_ws.assert_not_called()
    assert result["total_findings"] == 2
    assert result["safety_review_required"] is True
    severities = {f["severity"] for f in result["drug_drug_findings"] + result["drug_lab_findings"]}
    assert "High" in severities


@_demo_mode()
def test_check_pre_add_reports_no_coverage_for_unlisted_drug():
    context = {"candidate_name": "atorvastatin", "candidate_ingredients": [], "current_medications": []}
    with patch("app.services.databricks_client.WorkspaceClient") as mock_ws:
        result = check_pre_add(candidate_rxcui="83367", context=context, labs=None)

    mock_ws.assert_not_called()
    assert result["total_findings"] == 0
    assert "outside DEMO_MODE" in result["coverage_note"]


@_demo_mode()
def test_execute_sql_never_touches_workspace_client():
    demo_store.reset()
    with patch("app.services.sql_client.WorkspaceClient") as mock_ws:
        rows = execute_sql("SELECT 'OK' AS sql_status", expect_rows=True)

    mock_ws.assert_not_called()
    assert rows == [{"sql_status": "OK"}]


@_demo_mode()
def test_execute_sql_demo_store_insert_and_list_round_trip():
    demo_store.reset()
    execute_sql(
        """
        INSERT INTO healthcare.medication_safety.patient_medications
        (medication_record_id, patient_id, selected_rxcui, selected_name,
         primary_ingredient_rxcui, primary_ingredient_name, ingredients_json,
         strength, route, dose_form, term_type, medication_type, medication_status,
         source, is_active, added_at, updated_at)
        VALUES
        (:medication_record_id, :patient_id, :selected_rxcui, :selected_name,
         :primary_ingredient_rxcui, :primary_ingredient_name, :ingredients_json,
         :strength, :route, :dose_form, :term_type, :medication_type, :medication_status,
         :source, true, current_timestamp(), current_timestamp())
        """,
        parameters={
            "medication_record_id": "demo-1",
            "patient_id": "P001",
            "selected_rxcui": "29046",
            "selected_name": "lisinopril",
            "primary_ingredient_rxcui": "29046",
            "primary_ingredient_name": "lisinopril",
            "ingredients_json": "[]",
            "strength": None,
            "route": None,
            "dose_form": None,
            "term_type": "IN",
            "medication_type": "RX",
            "medication_status": "CURRENT",
            "source": "NLM RxNorm + RxTerms",
        },
    )

    rows = execute_sql(
        """
        SELECT medication_record_id, patient_id, selected_rxcui, selected_name,
               primary_ingredient_rxcui, primary_ingredient_name, ingredients_json,
               strength, route, dose_form, term_type, medication_type, medication_status,
               source, added_at, updated_at
        FROM healthcare.medication_safety.patient_medications
        WHERE patient_id = :patient_id AND is_active = true
        ORDER BY added_at
        """,
        parameters={"patient_id": "P001"},
        expect_rows=True,
    )

    assert len(rows) == 1
    assert rows[0]["medication_record_id"] == "demo-1"
    demo_store.reset()
