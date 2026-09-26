from unittest.mock import patch

import pytest

from app.services.databricks_client import DatabricksClientError
from app.services.medication_management import (
    MedicationManagementError,
    add_patient_medication,
    pre_add_check,
    remove_patient_medication,
)
from app.services.sql_client import SqlClientError


@patch("app.services.medication_management.execute_sql")
@patch("app.services.medication_management.get_medication_details")
def test_add_patient_medication_returns_duplicate_without_persisting(mock_details, mock_sql):
    mock_details.return_value = {"selected_name": "lisinopril", "ingredients": []}
    mock_sql.return_value = [
        {"medication_record_id": "existing-id", "selected_name": "lisinopril", "medication_status": "CURRENT"}
    ]

    result = add_patient_medication("P001", "29046", "PROPOSED", "RX", None)

    assert result["duplicate"] is True
    assert result["added"] is False
    # Only the duplicate-check SELECT should have run -- no INSERT for a duplicate.
    mock_sql.assert_called_once()


@patch("app.services.medication_management._persist_auto_discovered_rule")
@patch("app.services.medication_management.check_pre_add")
@patch("app.services.medication_management.execute_sql")
@patch("app.services.medication_management.get_medication_details")
def test_add_patient_medication_checks_safety_before_persisting(mock_details, mock_sql, mock_check, mock_persist_rule):
    mock_details.return_value = {
        "selected_name": "gemfibrozil", "ingredients": [{"rxcui": "4719", "name": "gemfibrozil"}],
        "primary_ingredient_rxcui": "4719", "primary_ingredient_name": "gemfibrozil",
        "strength": None, "route": None, "rxnorm_dose_form": None, "rxterms_dose_form": None,
        "term_type": "SCDF", "source": "NLM RxNorm + RxTerms",
    }

    call_order = []
    mock_check.side_effect = lambda **kw: call_order.append("safety_check") or {
        "drug_drug_findings": [], "drug_lab_findings": [], "total_findings": 0,
        "safety_review_required": False, "message": "No findings",
    }

    def sql_side_effect(statement, parameters=None, expect_rows=False):
        if "INSERT INTO" in statement and "patient_medications" in statement:
            call_order.append("insert")
            return []
        return []

    mock_sql.side_effect = sql_side_effect

    result = add_patient_medication("P001", "372300", "CURRENT", "RX", {"egfr": 55, "potassium": 5.1})

    assert result["added"] is True
    assert call_order == ["safety_check", "insert"], "safety check must run before persistence"
    mock_persist_rule.assert_not_called()


@patch("app.services.medication_management._persist_auto_discovered_rule")
@patch("app.services.medication_management.check_pre_add")
@patch("app.services.medication_management.execute_sql")
@patch("app.services.medication_management.get_medication_details")
def test_add_patient_medication_persists_auto_discovered_rule(mock_details, mock_sql, mock_check, mock_persist_rule):
    mock_details.return_value = {
        "selected_name": "gemfibrozil", "ingredients": [{"rxcui": "4719", "name": "gemfibrozil"}],
        "primary_ingredient_rxcui": "4719", "primary_ingredient_name": "gemfibrozil",
        "strength": None, "route": None, "rxnorm_dose_form": None, "rxterms_dose_form": None,
        "term_type": "SCDF", "source": "NLM RxNorm + RxTerms",
    }
    mock_sql.return_value = []
    finding = {
        "review_status": "Auto-discovered, unreviewed", "severity": "Unreviewed",
        "evidence_id": "FDA-4719-83367", "drug_a": "gemfibrozil", "drug_b": "atorvastatin",
    }
    mock_check.return_value = {
        "drug_drug_findings": [finding], "drug_lab_findings": [], "total_findings": 1,
        "safety_review_required": True, "message": "Unreviewed finding",
    }

    add_patient_medication("P001", "372300", "PROPOSED", "RX", {"egfr": 55, "potassium": 5.1})

    mock_persist_rule.assert_called_once_with(finding)


@patch("app.services.medication_management.check_pre_add")
@patch("app.services.medication_management.get_medication_details")
@patch("app.services.medication_management._current_medications_context")
def test_pre_add_check_wraps_databricks_error(mock_context, mock_details, mock_check):
    mock_context.return_value = []
    mock_details.return_value = {"selected_name": "gemfibrozil", "ingredients": []}
    mock_check.side_effect = DatabricksClientError("endpoint call failed")

    with pytest.raises(MedicationManagementError, match="endpoint call failed"):
        pre_add_check("P001", "372300", None)


@patch("app.services.medication_management.execute_sql")
def test_remove_patient_medication_soft_deletes(mock_sql):
    mock_sql.return_value = []

    result = remove_patient_medication("P001", "record-123")

    assert result["removed"] is True
    statement = mock_sql.call_args.args[0] if mock_sql.call_args.args else mock_sql.call_args.kwargs["statement"]
    assert "UPDATE" in statement
    assert "is_active = false" in statement


@patch("app.services.medication_management.execute_sql")
def test_remove_patient_medication_requires_ids(mock_sql):
    with pytest.raises(MedicationManagementError):
        remove_patient_medication("", "record-123")
    with pytest.raises(MedicationManagementError):
        remove_patient_medication("P001", "")
    mock_sql.assert_not_called()


@patch("app.services.medication_management.get_medication_details")
def test_add_patient_medication_rejects_invalid_status(mock_details):
    with pytest.raises(MedicationManagementError, match="medication_status"):
        add_patient_medication("P001", "372300", "BOGUS", "RX", None)
    mock_details.assert_not_called()
