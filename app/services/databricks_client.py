"""Thin client for the deployed medication-safety-agent Model Serving endpoint.

All Databricks-specific error handling lives here so the API layer only ever
sees DatabricksClientError — a single integration boundary that isolates
provider-specific error mapping in one place.

medication-lookup-api was retired — see docs/databricks-endpoint-lifecycle.md.
Its deterministic CRUD/lookup operations now live in
app/services/{sql_client,rxnorm_client,medication_management}.py; the one
genuinely clinical operation it had (pre_add_check) was folded into
medication-safety-agent as an action — see check_pre_add() below.
"""

import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

from app.config import settings


class DatabricksClientError(Exception):
    """Raised when the serving endpoint can't be reached or returns something unusable."""


def _client() -> WorkspaceClient:
    try:
        return WorkspaceClient(profile=settings.databricks_profile)
    except Exception as exc:
        raise DatabricksClientError(
            f"Could not authenticate with Databricks OAuth profile '{settings.databricks_profile}': {exc}"
        ) from exc


def analyze_patient(patient: dict) -> dict:
    """Send one patient through the medication-safety-agent endpoint and return its structured report."""
    if settings.demo_mode:
        from app.services.demo_fixtures import demo_analyze

        return demo_analyze(patient)

    client = _client()
    try:
        response = client.serving_endpoints.query(
            name=settings.serving_endpoint_name,
            dataframe_records=[patient],
        )
    except DatabricksError as exc:
        raise DatabricksClientError(f"Serving endpoint '{settings.serving_endpoint_name}' call failed: {exc}") from exc

    predictions = response.predictions
    if predictions is None:
        raise DatabricksClientError("Serving endpoint returned no predictions")
    # The model returns a single dict directly for a single-record request
    # (not a one-item list) — see docs/databricks-deployment-status.md.
    return predictions[0] if isinstance(predictions, list) else predictions


def check_pre_add(candidate_rxcui: str, context: dict, labs: dict | None) -> dict:
    """Run the clinical pre-add safety check via medication-safety-agent's
    pre_add_check action.

    Stateless: the caller (medication_management.py, which owns
    patient_medications) passes the candidate's ingredients and the
    patient's current medications directly in `context` rather than this
    endpoint querying Unity Catalog itself — matching how the ordinary
    analyze request already works. Backward compatible: this uses the
    `action`/`context_json` fields, which are optional and ignored by the
    ordinary flat-patient-dict analyze path used elsewhere in this module.
    """
    if settings.demo_mode:
        from app.services.demo_fixtures import demo_pre_add_check

        return demo_pre_add_check(candidate_rxcui, context, labs)

    record: dict = {
        "action": "pre_add_check",
        "candidate_rxcui": candidate_rxcui,
        "context_json": json.dumps(context),
    }
    if labs:
        record["labs"] = labs

    client = _client()
    try:
        response = client.serving_endpoints.query(
            name=settings.serving_endpoint_name,
            dataframe_records=[record],
        )
    except DatabricksError as exc:
        raise DatabricksClientError(f"Serving endpoint '{settings.serving_endpoint_name}' pre_add_check call failed: {exc}") from exc

    predictions = response.predictions
    if predictions is None:
        raise DatabricksClientError("Serving endpoint returned no predictions for pre_add_check")
    return predictions[0] if isinstance(predictions, list) else predictions
