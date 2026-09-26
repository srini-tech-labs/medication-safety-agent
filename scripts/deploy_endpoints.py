"""Idempotent Databricks Model Serving endpoint deployment utility.

Keeps this application's serving endpoint *names* stable and permanent.
For each configured endpoint:

  - missing                          -> CREATE it, pointing at the approved model version
  - exists, serving the approved version -> NO_OP
  - exists, serving a different version of the SAME model -> UPDATE to the approved version
  - exists, serving a DIFFERENT model, or in an unexpected shape -> FAIL_CLOSED (never guess)

This tool never renames or deletes an endpoint, and never registers a new model
version itself — it only ever points a stable endpoint name at a version that
already exists and is READY in Unity Catalog. Upgrading model logic is a
separate step (register a new version some other way), after which re-running
this tool will UPDATE the existing endpoint to that new approved version.

The approved target version is never hard-coded in this file — it has drifted
stale before (this file once pinned "4" long after the live endpoint had moved
on to v7). It must be supplied explicitly, one of:

  --target-version 7                          (CLI flag, highest precedence)
  MEDICATION_SAFETY_AGENT_TARGET_VERSION=7     (environment variable)
  --resolve-latest-ready                       (query Unity Catalog at runtime
                                                 for the highest READY version
                                                 of the registered model)

--resolve-latest-ready is opt-in, not the default: "latest registered" is not
the same thing as "clinically approved" — this project has previously hit a
retry-storm that produced orphaned extra registered versions (see
docs/databricks-deployment-status.md), so blindly deploying the newest
technically-READY version is a real risk, not a hypothetical one. Prefer an
explicit --target-version once you know which version you mean.

Usage:
    uv run python scripts/deploy_endpoints.py --target-version 7                 # dry run
    uv run python scripts/deploy_endpoints.py --target-version 7 --apply         # apply
    uv run python scripts/deploy_endpoints.py --target-version 7 --apply --wait  # apply, then block until READY
    uv run python scripts/deploy_endpoints.py --resolve-latest-ready             # dry run against latest READY
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.errors import NotFound  # noqa: E402
from databricks.sdk.service.serving import (  # noqa: E402
    EndpointCoreConfigInput,
    EndpointStateReady,
    ServedEntityInput,
)

from app.config import settings  # noqa: E402

# ============================================================
# CONFIGURED STABLE ENDPOINTS
# The single source of truth for "what should be deployed where" — except
# for the target version, which is deliberately NOT configured here (see
# module docstring). Endpoint names here must match app/config.py's
# settings exactly — the application only ever knows these names, never a
# model/version.
# ============================================================

WORKLOAD_SIZE = "Small"
SCALE_TO_ZERO_ENABLED = True

TARGET_VERSION_ENV_VAR = "MEDICATION_SAFETY_AGENT_TARGET_VERSION"


@dataclasses.dataclass(frozen=True)
class EndpointTarget:
    endpoint_name: str
    registered_model: str
    workload_size: str = WORKLOAD_SIZE
    scale_to_zero_enabled: bool = SCALE_TO_ZERO_ENABLED


ENDPOINT_TARGETS: list[EndpointTarget] = [
    EndpointTarget(
        endpoint_name=settings.serving_endpoint_name,
        registered_model="healthcare.medication_safety.medication_safety_agent",
    ),
    # medication-lookup-api is retired — its deterministic CRUD/lookup now
    # runs locally in app/services/medication_management.py, and its one
    # clinical operation (pre_add_check) was folded into
    # medication-safety-agent. Do not add a deployment target for it here —
    # see docs/databricks-endpoint-lifecycle.md.
]


@dataclasses.dataclass(frozen=True)
class EndpointSpec:
    endpoint_name: str
    registered_model: str
    target_version: str
    workload_size: str = WORKLOAD_SIZE
    scale_to_zero_enabled: bool = SCALE_TO_ZERO_ENABLED


def resolve_latest_ready_version(client: WorkspaceClient, registered_model: str) -> str:
    """Return the highest READY version number for a Unity Catalog registered model.

    Not a substitute for knowing which version you mean — see module
    docstring. Raises if the model has no READY version at all.
    """
    versions = list(client.model_versions.list(full_name=registered_model))
    ready = [v for v in versions if str(getattr(v, "status", "")).upper().endswith("READY")]
    if not ready:
        raise RuntimeError(f"No READY version found for registered model '{registered_model}'.")
    return str(max(int(v.version) for v in ready))


def build_endpoint_specs(
    client: WorkspaceClient,
    target_version: str | None,
    resolve_latest_ready: bool,
) -> list[EndpointSpec]:
    if target_version and resolve_latest_ready:
        raise ValueError("Pass either --target-version or --resolve-latest-ready, not both.")
    if not target_version and not resolve_latest_ready:
        raise ValueError(
            "No target version given. Pass --target-version, set "
            f"{TARGET_VERSION_ENV_VAR}, or pass --resolve-latest-ready."
        )

    specs = []
    for target in ENDPOINT_TARGETS:
        version = target_version or resolve_latest_ready_version(client, target.registered_model)
        specs.append(
            EndpointSpec(
                endpoint_name=target.endpoint_name,
                registered_model=target.registered_model,
                target_version=version,
                workload_size=target.workload_size,
                scale_to_zero_enabled=target.scale_to_zero_enabled,
            )
        )
    return specs


# ============================================================
# PLANNING (pure, read-only — safe to call from a dry run or a test)
# ============================================================


@dataclasses.dataclass(frozen=True)
class EndpointPlan:
    spec: EndpointSpec
    action: str  # "CREATE" | "UPDATE" | "NO_OP" | "FAIL_CLOSED"
    reason: str
    current_model: str | None = None
    current_version: str | None = None


def plan_for_endpoint(client: WorkspaceClient, spec: EndpointSpec) -> EndpointPlan:
    try:
        endpoint = client.serving_endpoints.get(name=spec.endpoint_name)
    except NotFound:
        return EndpointPlan(spec, "CREATE", f"Endpoint '{spec.endpoint_name}' does not exist yet.")
    except Exception as exc:  # noqa: BLE001 - deliberately fail closed on anything unexpected
        return EndpointPlan(spec, "FAIL_CLOSED", f"Could not read endpoint state: {exc}")

    served = endpoint.config.served_entities if endpoint.config else None
    if not served:
        return EndpointPlan(
            spec, "FAIL_CLOSED", f"Endpoint '{spec.endpoint_name}' exists but has no served entities — refusing to guess."
        )
    if len(served) != 1:
        return EndpointPlan(
            spec,
            "FAIL_CLOSED",
            f"Endpoint '{spec.endpoint_name}' serves {len(served)} entities — refusing to guess which one to manage.",
        )

    current = served[0]
    if current.entity_name != spec.registered_model:
        return EndpointPlan(
            spec,
            "FAIL_CLOSED",
            f"Endpoint '{spec.endpoint_name}' serves a DIFFERENT model ('{current.entity_name}') than expected "
            f"('{spec.registered_model}') — refusing to overwrite.",
            current_model=current.entity_name,
            current_version=current.entity_version,
        )

    if current.entity_version == spec.target_version:
        return EndpointPlan(
            spec,
            "NO_OP",
            f"Already serving {spec.registered_model} v{spec.target_version}.",
            current_model=current.entity_name,
            current_version=current.entity_version,
        )

    return EndpointPlan(
        spec,
        "UPDATE",
        f"Currently serving v{current.entity_version}, approved target is v{spec.target_version}.",
        current_model=current.entity_name,
        current_version=current.entity_version,
    )


# ============================================================
# APPLYING (the only functions that mutate anything)
# ============================================================


def _served_entities_for(spec: EndpointSpec) -> list[ServedEntityInput]:
    return [
        ServedEntityInput(
            name=f"{spec.endpoint_name}-{spec.target_version}",
            entity_name=spec.registered_model,
            entity_version=spec.target_version,
            workload_size=spec.workload_size,
            scale_to_zero_enabled=spec.scale_to_zero_enabled,
        )
    ]


def apply_plan(client: WorkspaceClient, plan: EndpointPlan) -> None:
    if plan.action == "CREATE":
        client.serving_endpoints.create(
            name=plan.spec.endpoint_name,
            config=EndpointCoreConfigInput(
                name=plan.spec.endpoint_name,
                served_entities=_served_entities_for(plan.spec),
            ),
        )
    elif plan.action == "UPDATE":
        client.serving_endpoints.update_config(
            name=plan.spec.endpoint_name,
            served_entities=_served_entities_for(plan.spec),
        )
    else:
        raise ValueError(f"apply_plan called on a non-actionable plan: {plan.action}")


def wait_until_ready(client: WorkspaceClient, endpoint_name: str, timeout_seconds: int = 900) -> None:
    deadline = time.time() + timeout_seconds
    while True:
        endpoint = client.serving_endpoints.get(name=endpoint_name)
        if endpoint.state and endpoint.state.ready == EndpointStateReady.READY:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Endpoint '{endpoint_name}' did not become READY within {timeout_seconds}s.")
        time.sleep(15)


# ============================================================
# REPORTING
# ============================================================


def format_plan(plan: EndpointPlan) -> str:
    lines = [
        f"Endpoint:        {plan.spec.endpoint_name}",
        f"UC model:        {plan.spec.registered_model}",
        f"Approved version: {plan.spec.target_version}",
        f"Workload size:   {plan.spec.workload_size}",
        f"Scale to zero:   {plan.spec.scale_to_zero_enabled}",
        f"Currently:       {'does not exist' if plan.current_version is None and plan.action == 'CREATE' else f'{plan.current_model} v{plan.current_version}'}",
        f"ACTION:          {plan.action}",
        f"Reason:          {plan.reason}",
    ]
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--target-version",
        default=os.environ.get(TARGET_VERSION_ENV_VAR),
        help=f"Approved model version to deploy. Overrides {TARGET_VERSION_ENV_VAR} if both are set.",
    )
    parser.add_argument(
        "--resolve-latest-ready",
        action="store_true",
        help=(
            "Resolve the highest READY registered-model version at runtime instead of an explicit "
            "--target-version. This is NOT the same as 'clinically approved' — see module docstring."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Actually create/update endpoints. Default is dry-run only.")
    parser.add_argument("--wait", action="store_true", help="With --apply, block until each changed endpoint is READY.")
    args = parser.parse_args()

    client = WorkspaceClient(profile=settings.databricks_profile)

    try:
        specs = build_endpoint_specs(client, args.target_version, args.resolve_latest_ready)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error() already exits

    if args.resolve_latest_ready:
        print(
            "--resolve-latest-ready was used: the version(s) below are the highest READY registered "
            "version, not a version anyone has necessarily reviewed. Confirm before --apply.\n"
        )

    plans = [plan_for_endpoint(client, spec) for spec in specs]

    print(f"{'DRY RUN' if not args.apply else 'APPLY'} — {len(plans)} configured endpoint(s)\n")
    for plan in plans:
        print(format_plan(plan))
        print()

    if any(p.action == "FAIL_CLOSED" for p in plans):
        print("One or more endpoints failed closed — stopping. No changes made.")
        return 1

    if not args.apply:
        print("Dry run only — no changes made. Re-run with --apply to execute the plan above.")
        return 0

    for plan in plans:
        if plan.action == "NO_OP":
            continue
        print(f"Applying {plan.action} for '{plan.spec.endpoint_name}'...")
        apply_plan(client, plan)
        if args.wait:
            print(f"Waiting for '{plan.spec.endpoint_name}' to become READY...")
            wait_until_ready(client, plan.spec.endpoint_name)
            print(f"'{plan.spec.endpoint_name}' is READY.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
