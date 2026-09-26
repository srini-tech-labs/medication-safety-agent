# Databricks endpoint lifecycle

## Architecture (as of the hybrid migration, 2026-09-24)

One application, one shared patient selector, two UI tabs:

- **Medication Safety Analysis** — calls the `medication-safety-agent` Model Serving endpoint (clinical intelligence: drug-drug/drug-lab rule evaluation, LLM explanation).
- **Medication Management** — deterministic CRUD/lookup (search, details, list, add-persistence, discontinue) runs **locally in FastAPI** (`app/services/medication_management.py`, `rxnorm_client.py`, `sql_client.py`), talking directly to RxNorm and to Unity Catalog via the SQL Statement Execution API. The one genuinely clinical piece of that workflow — the pre-add safety check — still runs centrally in Databricks, via a `pre_add_check` action added to `medication-safety-agent` (not a separate endpoint).

```
Browser
  └─ FastAPI
       ├─ Medication Safety Analysis  ──► medication-safety-agent (analyze)
       └─ Medication Management
            ├─ search / details / list / persistence ──► Unity Catalog SQL + RxNorm (local)
            └─ pre-add safety check                    ──► medication-safety-agent (pre_add_check action)
```

This keeps exactly one custom Model Serving endpoint for this application, instead of two.

## Why medication-lookup-api was retired

Its workload was primarily deterministic lookup/CRUD (health check, RxNorm search/details, `patient_medications` reads/writes) rather than model inference — 6 of its 7 actions never touched a model at all, and the 7th (`pre_add_check`) only conditionally called a *foundation-model* endpoint (not custom) for an openFDA-excerpt paraphrase. Model Serving was being used there as a generic authenticated-CRUD proxy, not for inference. It also happened to compete for the two-custom-endpoint limit observed in the Free Edition workspace used for this prototype, alongside an unrelated project's endpoint.

Its registered model, `healthcare.medication_safety.medication_lookup_pyfunc`, is **kept intact** in Unity Catalog as historical/recovery material — nothing was deleted. Only the serving endpoint and this app's runtime dependency on it were removed.

## The pre_add_check action

`medication-safety-agent`'s `predict()` dispatches on an optional `action` field:
- No `action` (or anything other than `"pre_add_check"`) → today's unchanged flat-patient-dict analyze behavior.
- `action == "pre_add_check"` → drug-drug/drug-lab rule matching against the **same baked-in rule snapshot** the analyze path already uses, with a live openFDA fallback (+ LLM paraphrase via the shared foundation-model endpoint) when the static tables have nothing for a pair.

It's stateless by design, matching the existing analyze path: FastAPI (which now owns `patient_medications`) passes the candidate's ingredients and the patient's current medications directly in the request rather than this endpoint querying Unity Catalog itself. This needed **no new Unity Catalog grants** — the model's rule data was already baked in at log time, with zero live SQL access.

**Known limitation**: when a user proceeds with an add despite an "Unreviewed" openFDA-discovered finding, FastAPI persists the learned rule into `drug_interactions`/`source_evidence` directly (SQL). Because `medication-safety-agent`'s rule snapshot is baked in at *deploy* time, not queried live, that newly-learned rule won't speed up subsequent checks for the same pair until the model is next redeployed — every check in between will re-run the openFDA fallback. This is a real tradeoff of moving persistence out of a live-querying endpoint; the old medication-lookup-api picked up a newly-learned rule within its ~5-minute in-memory cache window instead.

## Two lifecycle scenarios

**Endpoint restoration** (the endpoint was reclaimed/deleted, but the registered model still exists and is READY):
→ Recreate the endpoint only, pointing at the existing approved model version. Do **not** re-log/re-register the model — nothing about the model changed.

**Model logic upgrade** (code changes to the rule engine, the pre_add_check logic, etc.):
→ Register a new Unity Catalog model version → update the *existing* stable endpoint to serve it. The application's endpoint name (`medication-safety-agent`) never changes, so no app configuration change is needed on either side of this.

In both cases, use `scripts/deploy_endpoints.py` (dry-run by default, `--apply` to execute) — it's idempotent and safe to rerun: missing → create, correct version already served → no-op, wrong version served → update. It will not delete/recreate an endpoint merely because the model version changed, and it never registers a model version itself.

**After every redeploy of `medication-safety-agent`**, the served model gets a brand-new service-principal identity for Unity Catalog credential vending (confirmed repeatedly across this project's history) — but since this model has no live SQL/table dependency, no grants need to be reapplied for it specifically. This *does* still apply to any future work that gives it live table access.

## Open item

`medication-lookup-api`'s registered model (`healthcare.medication_safety.medication_lookup_pyfunc`) remains in Unity Catalog, unused, pending a decision on whether to delete it once this migration has been fully validated in production use.
