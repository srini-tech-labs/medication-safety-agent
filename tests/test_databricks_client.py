from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.errors import DatabricksError

from app.services.databricks_client import DatabricksClientError, analyze_patient

PATIENT = {"patient_id": "P003", "age": 72, "medications": ["Ibuprofen"]}


@patch("app.services.databricks_client.WorkspaceClient")
def test_analyze_patient_unwraps_single_dict_prediction(mock_ws_client):
    # The deployed model returns a bare dict (not a list) for a single-record
    # request — see docs/databricks-deployment-status.md.
    mock_client = MagicMock()
    mock_client.serving_endpoints.query.return_value.predictions = {"overall_risk": "LOW"}
    mock_ws_client.return_value = mock_client

    result = analyze_patient(PATIENT)

    assert result == {"overall_risk": "LOW"}


@patch("app.services.databricks_client.WorkspaceClient")
def test_analyze_patient_unwraps_list_prediction(mock_ws_client):
    mock_client = MagicMock()
    mock_client.serving_endpoints.query.return_value.predictions = [{"overall_risk": "HIGH"}]
    mock_ws_client.return_value = mock_client

    result = analyze_patient(PATIENT)

    assert result == {"overall_risk": "HIGH"}


@patch("app.services.databricks_client.WorkspaceClient")
def test_analyze_patient_wraps_databricks_error(mock_ws_client):
    mock_client = MagicMock()
    mock_client.serving_endpoints.query.side_effect = DatabricksError("endpoint cold-starting")
    mock_ws_client.return_value = mock_client

    with pytest.raises(DatabricksClientError, match="endpoint cold-starting"):
        analyze_patient(PATIENT)


@patch("app.services.databricks_client.WorkspaceClient")
def test_analyze_patient_raises_on_missing_predictions(mock_ws_client):
    mock_client = MagicMock()
    mock_client.serving_endpoints.query.return_value.predictions = None
    mock_ws_client.return_value = mock_client

    with pytest.raises(DatabricksClientError, match="no predictions"):
        analyze_patient(PATIENT)
