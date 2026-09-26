# Databricks notebooks (exported snapshots)

These are exported copies of the notebooks that live in the Databricks workspace at
`/Workspace/Healthcare AI Agents/Medication Safety Intelligence Agent/`. In the current
architecture, only one of them backs an active Model Serving endpoint — see
`../docs/databricks-agent-notes.md` and `../docs/databricks-deployment-status.md` for the full
history.

**This is not a live sync.** Databricks Git folders didn't support this workspace's git
provider, so these files are manually exported/imported rather than continuously synced.

- `01-medication-safety-agent.py` — main engine: RxNorm normalization, drug-drug/drug-lab rule
  evaluation, LLM explanation step, and the `pre_add_check` action used by Medication Management.
  Backs the one active endpoint, `medication-safety-agent`.
- `02-drug-data-integration.py` — RxNorm/DailyMed integration and rule/evidence table population.
- `medication-lookup-pyfunc-notebook.py` — **retired, historical only** (see the notice at the top
  of that file). It backed a second endpoint, `medication-lookup-api`, whose deterministic
  CRUD/lookup responsibilities now run locally in `app/services/medication_management.py`; its one
  genuinely clinical operation (the pre-add safety check) was folded into
  `medication-safety-agent`. See `../docs/databricks-endpoint-lifecycle.md` for why.

## Updating

To pull the latest from Databricks:
```
databricks workspace export "/Workspace/Healthcare AI Agents/Medication Safety Intelligence Agent/<notebook name>" \
  --format SOURCE --profile medication-safety-agent > databricks/<local-file>.py
```

To push a local edit back and redeploy (for `01-medication-safety-agent.py`, which re-registers a
model version; run `scripts/deploy_endpoints.py` afterward to point the endpoint at that new
version — see `../docs/databricks-endpoint-lifecycle.md`):
```
databricks workspace import "/Workspace/Healthcare AI Agents/Medication Safety Intelligence Agent/<notebook name>" \
  --file databricks/<local-file>.py --format SOURCE --language PYTHON --overwrite --profile medication-safety-agent

databricks jobs submit --profile medication-safety-agent --no-wait --json '{
  "run_name": "redeploy",
  "tasks": [{"task_key": "run_notebook", "notebook_task": {"notebook_path": "/Workspace/Healthcare AI Agents/Medication Safety Intelligence Agent/<notebook name>"}}]
}'
```

`medication-safety-agent` has no live Unity Catalog access (its rule data is baked in at
model-logging time), so redeploying it needs no follow-up grants. That was not true of the retired
`medication-lookup-pyfunc-notebook.py`, which read/wrote several tables directly — each redeploy of
that notebook minted a new service-principal identity that needed fresh `GRANT SELECT, MODIFY`
statements, documented in `../docs/databricks-deployment-status.md` as historical context only.
