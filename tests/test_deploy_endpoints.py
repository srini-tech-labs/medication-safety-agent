from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound

from scripts.deploy_endpoints import (
    EndpointSpec,
    build_endpoint_specs,
    format_plan,
    plan_for_endpoint,
    resolve_latest_ready_version,
)

SPEC = EndpointSpec(
    endpoint_name="medication-safety-agent",
    registered_model="healthcare.medication_safety.medication_safety_agent",
    target_version="4",
)


class _FakeVersion:
    def __init__(self, version: int, status: str):
        self.version = version
        self.status = status  # code checks str(status).upper().endswith("READY"), matching the real enum's str()


def _served(entity_name, entity_version):
    served = MagicMock()
    served.entity_name = entity_name
    served.entity_version = entity_version
    return served


def test_plan_creates_when_endpoint_missing():
    client = MagicMock()
    client.serving_endpoints.get.side_effect = NotFound("does not exist")

    plan = plan_for_endpoint(client, SPEC)

    assert plan.action == "CREATE"
    assert "does not exist" in plan.reason


def test_plan_no_ops_when_already_on_approved_version():
    client = MagicMock()
    endpoint = MagicMock()
    endpoint.config.served_entities = [_served(SPEC.registered_model, SPEC.target_version)]
    client.serving_endpoints.get.return_value = endpoint

    plan = plan_for_endpoint(client, SPEC)

    assert plan.action == "NO_OP"


def test_plan_updates_when_serving_wrong_version_of_approved_model():
    client = MagicMock()
    endpoint = MagicMock()
    endpoint.config.served_entities = [_served(SPEC.registered_model, "3")]
    client.serving_endpoints.get.return_value = endpoint

    plan = plan_for_endpoint(client, SPEC)

    assert plan.action == "UPDATE"
    assert plan.current_version == "3"


def test_plan_fails_closed_when_serving_a_different_model():
    client = MagicMock()
    endpoint = MagicMock()
    endpoint.config.served_entities = [_served("healthcare.medication_safety.some_other_model", "1")]
    client.serving_endpoints.get.return_value = endpoint

    plan = plan_for_endpoint(client, SPEC)

    assert plan.action == "FAIL_CLOSED"
    assert "DIFFERENT model" in plan.reason


def test_plan_fails_closed_when_multiple_served_entities():
    client = MagicMock()
    endpoint = MagicMock()
    endpoint.config.served_entities = [
        _served(SPEC.registered_model, SPEC.target_version),
        _served(SPEC.registered_model, "3"),
    ]
    client.serving_endpoints.get.return_value = endpoint

    plan = plan_for_endpoint(client, SPEC)

    assert plan.action == "FAIL_CLOSED"
    assert "2 entities" in plan.reason


def test_plan_fails_closed_on_unexpected_error():
    client = MagicMock()
    client.serving_endpoints.get.side_effect = RuntimeError("transient network error")

    plan = plan_for_endpoint(client, SPEC)

    assert plan.action == "FAIL_CLOSED"
    assert "transient network error" in plan.reason


def test_format_plan_never_includes_a_token_or_credential():
    client = MagicMock()
    client.serving_endpoints.get.side_effect = NotFound("does not exist")
    plan = plan_for_endpoint(client, SPEC)

    output = format_plan(plan)

    for leaked in ("token", "Bearer", "dapi", "Authorization"):
        assert leaked not in output


# ============================================================
# Target-version resolution (no more hard-coded version pin)
# ============================================================


def test_resolve_latest_ready_version_picks_highest_ready_version():
    client = MagicMock()
    client.model_versions.list.return_value = [
        _FakeVersion(4, "ModelVersionInfoStatus.READY"),
        _FakeVersion(7, "ModelVersionInfoStatus.READY"),
        _FakeVersion(6, "ModelVersionInfoStatus.READY"),
    ]

    version = resolve_latest_ready_version(client, "healthcare.medication_safety.medication_safety_agent")

    assert version == "7"


def test_resolve_latest_ready_version_ignores_non_ready_versions():
    client = MagicMock()
    client.model_versions.list.return_value = [
        _FakeVersion(7, "ModelVersionInfoStatus.READY"),
        _FakeVersion(8, "ModelVersionInfoStatus.PENDING"),
    ]

    version = resolve_latest_ready_version(client, "healthcare.medication_safety.medication_safety_agent")

    assert version == "7"


def test_resolve_latest_ready_version_raises_when_none_ready():
    client = MagicMock()
    client.model_versions.list.return_value = [_FakeVersion(3, "ModelVersionInfoStatus.PENDING")]

    with pytest.raises(RuntimeError, match="No READY version"):
        resolve_latest_ready_version(client, "healthcare.medication_safety.medication_safety_agent")


def test_build_endpoint_specs_requires_a_version_source():
    client = MagicMock()

    with pytest.raises(ValueError, match="No target version given"):
        build_endpoint_specs(client, target_version=None, resolve_latest_ready=False)


def test_build_endpoint_specs_rejects_both_sources_at_once():
    client = MagicMock()

    with pytest.raises(ValueError, match="not both"):
        build_endpoint_specs(client, target_version="7", resolve_latest_ready=True)


def test_build_endpoint_specs_uses_explicit_target_version():
    client = MagicMock()

    specs = build_endpoint_specs(client, target_version="7", resolve_latest_ready=False)

    assert len(specs) == 1
    assert specs[0].target_version == "7"
    assert specs[0].registered_model == "healthcare.medication_safety.medication_safety_agent"
    client.model_versions.list.assert_not_called()


def test_build_endpoint_specs_resolves_latest_ready_when_requested():
    client = MagicMock()
    client.model_versions.list.return_value = [
        _FakeVersion(7, "ModelVersionInfoStatus.READY"),
        _FakeVersion(5, "ModelVersionInfoStatus.READY"),
    ]

    specs = build_endpoint_specs(client, target_version=None, resolve_latest_ready=True)

    assert specs[0].target_version == "7"
