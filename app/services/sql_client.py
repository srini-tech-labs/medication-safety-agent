"""Direct Unity Catalog SQL access for the Medication Management feature.

This runs server-side in FastAPI, using the same Databricks OAuth profile as
`databricks_client.py`. It replaces the old medication-lookup-api Model
Serving endpoint for deterministic CRUD/lookup — see
docs/databricks-endpoint-lifecycle.md for why that moved here instead of
staying behind a serving endpoint. Credentials and SQL access never reach
the browser; only this module talks to the SQL warehouse.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.sql import StatementParameterListItem

from app.config import settings


class SqlClientError(Exception):
    """Raised when a Unity Catalog SQL statement can't be executed."""


def _client() -> WorkspaceClient:
    try:
        return WorkspaceClient(profile=settings.databricks_profile)
    except Exception as exc:
        raise SqlClientError(
            f"Could not authenticate with Databricks OAuth profile '{settings.databricks_profile}': {exc}"
        ) from exc


def execute_sql(statement: str, parameters: dict | None = None, expect_rows: bool = False) -> list[dict]:
    """Run one parameterized SQL statement, returning rows as dicts (or [] for writes)."""
    if settings.demo_mode:
        from app.services.demo_store import demo_execute_sql

        return demo_execute_sql(statement, parameters, expect_rows)

    client = _client()
    try:
        response = client.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=settings.databricks_warehouse_id,
            parameters=[
                StatementParameterListItem(name=name, value=None if value is None else str(value))
                for name, value in (parameters or {}).items()
            ],
            wait_timeout="30s",
        )
    except DatabricksError as exc:
        raise SqlClientError(f"Databricks SQL failed: {exc}") from exc

    if response.status and response.status.error:
        raise SqlClientError(f"Databricks SQL failed: {response.status.error}")

    if not expect_rows:
        return []

    columns = [c.name for c in response.manifest.schema.columns] if response.manifest and response.manifest.schema else []
    data_array = response.result.data_array if response.result else None
    return [dict(zip(columns, row)) for row in (data_array or [])]
