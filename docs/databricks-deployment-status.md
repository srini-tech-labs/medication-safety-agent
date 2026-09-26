# Databricks Deployment Status

Carried over from the spike session (window retired 2026-09-08) so this context isn't lost. Records what has actually been *done* in the Databricks workspace, as opposed to `databricks-agent-notes.md` which describes the original prototype's architecture.

## 2026-09-24: outage, investigation, and hybrid architecture migration

On resuming work, the app failed with `Serving endpoint 'medication-safety-agent' call failed: The given endpoint does not exist`. Investigation found **both** custom serving endpoints (`medication-safety-agent` and `medication-lookup-api`) missing, while both underlying Unity Catalog registered models remained intact and `READY`. `system.access.audit` showed no `deleteServingEndpoint` event for either endpoint in the available window — the cause is not provable from the data on hand; the endpoints were simply gone with the models untouched, and restoration only ever required recreating the endpoints, not rebuilding anything.

This was used as an opportunity to also resolve the two-custom-endpoint limit observed in the Free Edition workspace used for this prototype (already fully consumed by `medication-safety-agent` and an unrelated project's `energy-customer-insights`) by migrating to a **hybrid architecture**: `medication-lookup-api` was retired, its deterministic CRUD/lookup moved into FastAPI, and its one clinical operation (`pre_add_check`) was folded into `medication-safety-agent` as an additional action. See `docs/databricks-endpoint-lifecycle.md` for the full architecture, rationale, and redeploy procedure.

**Real bugs hit and fixed while adding `pre_add_check` to `medication-safety-agent`** (each confirmed via a local reproduction harness before redeploying, to stop guessing blind at Databricks-side errors):
1. `mlflow.pyfunc.log_model(input_example=<list of dicts with different keys per row>)` silently produces **no signature at all** (not even an error) — MLflow's implicit inference needs an explicit `pd.DataFrame(...)`, not a raw heterogeneous list, to correctly union mismatched columns as optional.
2. A pandas-sourced row's missing optional column comes back as `NaN` (a **truthy** float) when converted via `to_dict(orient="records")` — not `None`. Code using `x or default` to handle "field not provided" doesn't catch this and can call methods (`.items()`, etc.) on a bare float. Fixed by explicit `isinstance` checks at the row-extraction boundary instead of truthiness.
3. `log_model`'s own optional self-check of `input_example` against the computed `signature` is **stricter** than the signature's own "optional" designation for complex (Array/struct) types — it rejects `NaN` for those even though the signature says optional, an inconsistency between MLflow's inference and validation layers. Fixed by dropping `input_example=` entirely and passing only the explicitly-computed `signature=` — `input_example` is documentation-only for the MLflow UI, not required by Unity Catalog.

## Endpoint: deployed and working

- **Name**: `medication-safety-agent`
- **Historical checkpoint (2026-09-08)**: serving model version 3 (`served_model_name='medication_safety_agent-3'`) at the time this note was written. This file is a point-in-time log, not a live status page — for the version the endpoint is serving *right now*, check the endpoint directly (`databricks serving-endpoints get medication-safety-agent`) or run `scripts/deploy_endpoints.py` (dry-run mode reports the currently-served version without changing anything).
- **Registered model**: `healthcare.medication_safety.medication_safety_agent` (Unity Catalog)
- **Invocation**: `WorkspaceClient().serving_endpoints.query(name="medication-safety-agent", dataframe_records=[patient])` — the local app's `databricks_client.py` should use this exact call shape.
- **Smoke-tested against `patient_p003`**: returned `overall_risk: MODERATE`, 1 drug-drug + 1 drug-lab finding, correct evidence provenance, coherent AI explanation — matches the expected P003 result exactly.
- Free Edition **does** support custom Model Serving endpoints (was an open question — now resolved, no fallback needed).

## Fixes required to make the notebook logic servable (already applied)

1. **No Spark in the serving container**: `drug_interactions`/`lab_rules`/`evidence_lookup` were loaded via `spark.table(...).collect()` at notebook scope. Fixed by snapshotting them to plain Python data at model-logging time and baking the snapshot into the `MLflow pyfunc` model's constructor (not re-queried live in `predict()`).

2. **`Decimal` not JSON-serializable**: `drug_lab_rules.threshold` is a Unity Catalog `DECIMAL` column, which comes back as Python `decimal.Decimal` via `spark.table(...).collect()` / `row.asDict()`. MLflow's serving response serializer can't encode that (`Object of type Decimal is not JSON serializable`). Fixed by recursively converting `Decimal` → `float` on the rule/evidence snapshot before logging the model. **This is not a one-time data fix** — any future re-snapshot (e.g. after editing rules in the catalog) must repeat the conversion.

3. **Nested endpoint call**: the AI-explanation step calls `w.serving_endpoints.query(name="databricks-meta-llama-3-3-70b-instruct", ...)` from inside `predict()`. This required declaring `resources=[DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct")]` at `mlflow.pyfunc.log_model(...)` time so the wrapping endpoint can authenticate the nested call.

Reference shape of the logged model (for reproducing/updating it, not needed by the web app):

```python
class MedicationSafetyModel(PythonModel):
    def __init__(self, drug_interactions, lab_rules, evidence_lookup):
        self.drug_interactions = drug_interactions
        self.lab_rules = lab_rules
        self.evidence_lookup = evidence_lookup

    def predict(self, context, model_input):
        records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else model_input
        results = [
            medication_safety_api(p, self.drug_interactions, self.lab_rules, self.evidence_lookup)
            for p in records
        ]
        return results if len(results) > 1 else results[0]
```

## Verified / unverified risks

- **Verified working**: serving-container network egress to RxNorm's public API — the smoke test succeeded, and `run_medication_safety_agent_for_serving` calls `normalize_medication_list` → RxNorm live, so egress is not blocked.
- **Still unverified**: cold-start latency on `scale_to_zero` — the endpoint has been kept warm throughout testing so far, so a cold start hasn't actually been observed.

## Live-tested: the model's real input schema is stricter than the original notes

Discovered 2026-09-08 while wiring up the web app's first live request to the endpoint (not previously documented — the earlier notes described `labs` as a generic name→value map like `{K: 5.1, eGFR: 55}`; that is **not** what the deployed model actually accepts):

- The MLflow model signature was inferred from whatever example was passed at `log_model()` time, and locked in exactly that shape. The real signature is:
  ```
  patient_id: string (required)
  age: long (required)
  medications: Array(string) (required)
  conditions: Array(string) (required)      # not optional — must send [] if none
  labs: {egfr: long (required), potassium: double (required)}   # exactly these two keys, no others
  allergies: Any (required)                 # must be present — [] works
  ```
- `labs` is **not** a free-form lab dict — only `egfr` (must serialize as an integer, a float like `48.0` is rejected — `"Failed to enforce schema for key egfr. Expected type long, received type float"`) and `potassium` (a float) are accepted.
- `conditions` and `allergies` must be present in the request even when empty (`[]`) — omitting them fails schema validation (`"Model is missing inputs ['conditions', 'labs', 'allergies']"`).
- This app's `app/api/patients.py` `Labs` model and `app/web/routes.py` seed data were built against this confirmed real schema, not the original notes' generic description.
- **Implication for future rule types**: if new drug-lab rules reference labs other than eGFR/potassium, the model will need to be re-logged with an input example that includes those columns — the current endpoint cannot accept arbitrary lab names.

## Historical second endpoint: `medication-lookup-api` (2026-09-11)

- **Registered model**: `healthcare.medication_safety.medication_lookup_pyfunc` v2, served as `medication-lookup-2` under the `medication-lookup-api` endpoint. Backs the web app's new Medication Management feature (search/add/discontinue patient medications) — a separate concern from `medication-safety-agent`'s read-only drug-safety analysis.
- **Request contract differs from `medication-safety-agent`**: two DataFrame columns, `action` (string) and `payload_json` (a **JSON-encoded string** of the payload dict — call `json.dumps()` before sending, don't send a nested object). Response `predictions` unwraps (same bare-dict-vs-list quirk as the other endpoint) to a dict with a `result_json` **string** key that itself needs `json.loads()` — a double-JSON envelope, not a single layer like `medication-safety-agent`. The decoded envelope is always `{"ok": bool, "action": str, "result": dict | null, "error": {"type": str, "message": str} | null}`.
- **Backed by real Delta tables**, confirmed via `SHOW TABLES IN healthcare.medication_safety`: `patient_medications` (per-patient active/discontinued medication records — genuinely separate from `medication-safety-agent`'s hardcoded `P001-P003` seed data; a patient can have 3 seed medications for safety analysis and 0 rows in `patient_medications`) and `rxnorm_medication_cache` (write-through RxNorm/RxTerms lookup cache).

### Permission bug found and fixed (2026-09-11)

`get_medication_details`, `pre_add_check`, and `add_patient_medication` all failed live with `PERMISSION_DENIED: User does not have UPDATE/INSERT on Table ...` — the serving endpoint's runtime identity is a **service principal** (`00000000-0000-0000-0000-000000000001`), not the deploying user (`workspace-owner@example.com`). Model Serving's automatic Unity Catalog credential vending apparently only grants read access by default; anything the pyfunc writes to (the RxNorm cache, the patient medications table) needs an **explicit grant to the service principal itself** — granting to `` `account users` `` did **not** work (that group appears to cover human accounts only, not service principals; confirmed via SQL query history showing the SP as the actual `user_name` on the failing queries).

Fix applied directly via the SQL warehouse (`0000000000000000`):
```sql
GRANT SELECT, MODIFY ON TABLE healthcare.medication_safety.rxnorm_medication_cache TO `00000000-0000-0000-0000-000000000001`;
GRANT SELECT, MODIFY ON TABLE healthcare.medication_safety.patient_medications TO `00000000-0000-0000-0000-000000000001`;
```
All 7 actions confirmed working live after this. **If this model is ever re-deployed under a new served-model/service-principal identity, these grants will need to be reapplied** — check `SHOW GRANTS ON TABLE ...` and the SQL query history's `user_name` on any future `PERMISSION_DENIED` to find the current SP id.

### Confirmed request/response shapes for all 7 actions (live-tested 2026-09-11)

**`health`** — payload `{}` → `result: {"status": "OK", "rxnorm": {"version", "apiVersion"}, "databricks_sql": [{"sql_status": "OK"}], "service": "medication-lookup-api"}`

**`list_patient_medications`** — payload `{"patient_id"}` → `result: {"patient_id", "medications": [...], "count"}`. Each medication record: `medication_record_id, patient_id, selected_rxcui, selected_name, primary_ingredient_rxcui, primary_ingredient_name, strength, route, dose_form, term_type, medication_type, medication_status, source, added_at, updated_at, ingredients: [{rxcui, name}]`.

**`search_medications`** — payload `{"query", "limit"}` → `result: {"query", "matches": [{"rxcui", "name", "rank", "score", "search_scope"}, ...], "count", "cache", "source"}`.

**`get_medication_details`** — payload `{"rxcui"}` → `result: {"selected_rxcui", "selected_name", "rxnorm_name", "full_name", "full_generic_name", "strength", "route", "rxterms_dose_form", "rxnorm_dose_form", "term_type", "generic_rxcui", "primary_ingredient_rxcui", "primary_ingredient_name", "ingredients": [{rxcui, name}], "medication_type", "source", "cache"}`. Fields not known for a given medication come back as `null` (e.g. `strength`/`route` were `null` for lisinopril Oral Tablet, rxcui 372614) — the UI must omit missing fields, not render `None`/`null`.

**`pre_add_check`** — payload `{"patient_id", "rxcui", "labs": {"egfr", "potassium"}}` (send `{}` for `labs` if unavailable — never invent values) → `result: {"patient_id", "candidate_medication": <same shape as get_medication_details result>, "drug_drug_findings": [...], "drug_lab_findings": [...], "total_findings", "safety_review_required", "message", "coverage_note"}`.
- `drug_lab_findings[]` shape here is **different from `medication-safety-agent`'s** finding shape: `{"finding_type", "severity", "drug", "lab", "observed_value", "operator", "threshold", "concern", "threshold_basis", "evidence_id", "source"}` — note the message field is called `concern` here, not `message`, and evidence is flat (`evidence_id`/`source` directly) rather than a nested `evidence: {...}` object with `source_setid`/`section_title`/`subsection_title` like the other endpoint. Do not assume the two endpoints share a finding schema.
- `drug_drug_findings[]` shape unconfirmed (empty in every live test so far — P001 has no other current medications on file to interact with).

**`add_patient_medication`** — payload `{"patient_id", "rxcui", "medication_status": "CURRENT"|"PROPOSED", "medication_type": "RX"|"OTC"|"UNKNOWN", "labs"}`.
- Success: `result: {"added": true, "duplicate": false, "medication_record_id", "patient_id", "medication": <medication-details shape>, "medication_type", "medication_status", "safety": <same shape as pre_add_check's result — this is the authoritative post-add safety result, may differ from the pre-add preview>, "message"}`.
- Duplicate: `result: {"added": false, "duplicate": true, "patient_id", "selected_rxcui", "existing_record": {"medication_record_id", "selected_name", "medication_status"}, "message": "This medication is already active on the patient list."}` — `ok` is still `true`; a duplicate is not an error.

**`remove_patient_medication`** (discontinue) — payload `{"patient_id", "medication_record_id"}` → `result: {"removed": true, "patient_id", "medication_record_id", "message": "Medication marked DISCONTINUED and retained for history."}`. Confirmed live: the record disappears from `list_patient_medications` immediately after (soft delete, not a hard delete — matches the "preserve history" requirement, though history itself isn't exposed through any of these 7 actions).

## Dynamic DailyMed/FDA fallback signal (added 2026-09-11)

The static `drug_interactions` table only covers 3 hand-curated drug pairs (Lisinopril, Spironolactone, Ibuprofen). Real gap found live: checking atorvastatin (already Current for P001) against gemfibrozil returned `[NONE]` even though this is a well-established, clinically significant interaction (gemfibrozil inhibits statin metabolism, raising myopathy/rhabdomyolysis risk) — simply not in the curated table.

**Design** (agreed with the user before implementing): `pre_add_check`'s drug-drug screening now falls back to a live openFDA label lookup, but *only* when the static table finds nothing for a pair — the static table stays the fast/primary path. The fallback:
1. Fetches the candidate drug's openFDA label (`drug_interactions`/`warnings_and_cautions`/`contraindications` fields), and the existing drug's label as a second attempt if the first finds nothing (mentions aren't always symmetric).
2. If the other drug's name appears in one of those fields, the **verbatim excerpt is the fact** — nothing about severity or existence is inferred or guessed.
3. The `databricks-meta-llama-3-3-70b-instruct` LLM is called *only* to paraphrase that specific excerpt in plain language (explicitly instructed not to add claims beyond the excerpt) — same "rules determine facts, LLM only explains" principle as the main Medication Safety Agent, just applied to a label excerpt instead of a curated rule.
4. The finding gets `severity: "Unreviewed"` (a new tier, not High/Moderate/Low) and `review_status: "Auto-discovered, unreviewed"`, and `safety_review_required` is still set `true` so the UI's acknowledgement gate applies.
5. If the user proceeds with `add_patient_medication` despite (or informed by) this finding, the rule is persisted into `drug_interactions` (+ a matching `source_evidence` row with the real excerpt, `reviewed: false`) so **future checks for the same ingredient pair use the fast static path** instead of repeating the live FDA lookup — this is a self-reinforcing "just-in-time rule learning" loop, not a one-off.

**Scope note**: this fallback currently covers drug-**drug** findings only, not drug-lab. Drug-lab screening still only uses the static `drug_lab_rules` table.

**Real bugs found and fixed during this build**:
- Self-pair bug: an early version searched a drug's own label for its own name when checking a candidate whose ingredient RxCUI matched one already on file (e.g. checking a different pack of a drug the patient already takes), always "finding" a match against itself. Fixed with an explicit `new_rxcui == current_rxcui` skip.
- **Each re-logged model version gets a brand-new service-principal identity for automatic Unity Catalog credential vending** — confirmed by checking `SQL query history`'s `user_name` after each redeploy, which showed a different principal UUID every time the model was re-logged (v2→v4 alone used two different SPs). **Every redeploy of this model requires re-running the `GRANT SELECT, MODIFY` statements** (on `patient_medications`, `rxnorm_medication_cache`, `drug_interactions`, `drug_lab_rules`, `source_evidence`) **to whatever new principal shows up in query history** — this is not a one-time fix, it's a step in every future redeploy of this specific model.
- Duplicate-insert race: the in-memory rule cache (5-minute TTL per served-model replica) means a second add for the same pair can re-trigger the fallback-and-persist path on a replica that hasn't yet seen a rule another replica already wrote, before that replica's own cache would naturally refresh. Fixed by checking for the deterministic `evidence_id` (`FDA-<sorted ingredient rxcuis>`) in `drug_interactions` before inserting, skipping if already present.

**Local app changes**: `app/web/templates/index.html`'s drug-drug finding card now special-cases `review_status === "Auto-discovered, unreviewed"` — shows the trusted-source excerpt first (labeled "From FDA label (unreviewed)"), the AI explanation second, and a closing disclaimer to confirm with a pharmacist/clinician, per the user's explicit design ask. New `Unreviewed` badge/panel color (`--info`/`--info-soft` tokens) distinct from the High/Moderate/Low/None tiers.

## Local environment setup

Running any of this against a real workspace requires the `databricks` CLI and a working OAuth profile
(`databricks auth login`) matching `DATABRICKS_PROFILE` in `.env` — see the "Live mode" section of the
top-level [README](../README.md) for the full setup steps.
