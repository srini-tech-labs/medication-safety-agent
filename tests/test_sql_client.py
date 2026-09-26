from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.errors import DatabricksError

from app.services.sql_client import SqlClientError, execute_sql


@patch("app.services.sql_client.WorkspaceClient")
def test_execute_sql_returns_rows_as_dicts(mock_ws_client):
    mock_client = MagicMock()
    response = MagicMock()
    response.status.error = None
    col_a, col_b = MagicMock(), MagicMock()
    col_a.name = "patient_id"
    col_b.name = "medication_status"
    response.manifest.schema.columns = [col_a, col_b]
    response.result.data_array = [["P001", "CURRENT"]]
    mock_client.statement_execution.execute_statement.return_value = response
    mock_ws_client.return_value = mock_client

    rows = execute_sql("SELECT * FROM x", parameters={"a": "1"}, expect_rows=True)

    assert rows == [{"patient_id": "P001", "medication_status": "CURRENT"}]


@patch("app.services.sql_client.WorkspaceClient")
def test_execute_sql_returns_empty_list_when_not_expecting_rows(mock_ws_client):
    mock_client = MagicMock()
    response = MagicMock()
    response.status.error = None
    mock_client.statement_execution.execute_statement.return_value = response
    mock_ws_client.return_value = mock_client

    rows = execute_sql("UPDATE x SET y = 1")

    assert rows == []


@patch("app.services.sql_client.WorkspaceClient")
def test_execute_sql_raises_on_databricks_error(mock_ws_client):
    mock_client = MagicMock()
    mock_client.statement_execution.execute_statement.side_effect = DatabricksError("warehouse unavailable")
    mock_ws_client.return_value = mock_client

    with pytest.raises(SqlClientError, match="warehouse unavailable"):
        execute_sql("SELECT 1")


@patch("app.services.sql_client.WorkspaceClient")
def test_execute_sql_raises_on_statement_level_error(mock_ws_client):
    mock_client = MagicMock()
    response = MagicMock()
    response.status.error = "PERMISSION_DENIED"
    mock_client.statement_execution.execute_statement.return_value = response
    mock_ws_client.return_value = mock_client

    with pytest.raises(SqlClientError, match="PERMISSION_DENIED"):
        execute_sql("SELECT 1")
