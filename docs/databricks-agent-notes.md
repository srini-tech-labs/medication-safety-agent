# Databricks Agent — Architecture & Progress Notes

This is the record of what already exists in the Databricks workspace (`Healthcare AI Agents / Medication Safety Intelligence Agent`), carried over from the notebook-based prototyping session. The web app in this repo is built as a client of this agent — it does not reimplement this logic locally.

> Synthetic/fictional patient data only. Not for clinical use.

## Databricks workspace layout

```
Workspace
└── Healthcare AI Agents
    └── Medication Safety Intelligence Agent
        ├── 01 - Medication Safety Agent      (main engine notebook)
        └── 02 - Drug Data Integration        (RxNorm / DailyMed integration work)
```

## Catalog: `healthcare.medication_safety`

| Table | Purpose |
|---|---|
| `drug_interactions` | Executable drug-drug safety rules (keyed by RxCUI pairs) |
| `drug_lab_rules` | Executable drug-lab safety rules (drug RxCUI + lab name/operator/threshold) |
| `source_evidence` | DailyMed-sourced evidence backing the rules above, keyed by `evidence_id` |

Rules and evidence are deliberately separate tables — evidence never *becomes* a rule automatically, and a rule's `threshold_basis` field says explicitly when a prototype threshold (e.g. `eGFR < 60`) is not itself validated by the cited evidence, only the general clinical relationship is.

Current prototype coverage: Lisinopril, Spironolactone, Ibuprofen — 3 drug-drug rules, 3 drug-lab rules, 6 evidence records.

## RxNorm normalization (verified from notebook source)

```python
def get_rxcui(drug_name): ...       # rxnav.nlm.nih.gov/REST/rxcui.json lookup
def normalize_drug(drug_name): ...  # get_rxcui + /REST/rxcui/{rxcui}/properties.json
def normalize_medication_list(medications): ...
```

Confirmed RxCUIs: Lisinopril = 29046, Spironolactone = 9997, Ibuprofen = 5640.

## Core engine: `run_medication_safety_agent(patient)` (verified from notebook source)

1. Normalize the patient's medication list via RxNorm → RxCUIs.
2. Check every RxCUI pair (`itertools.combinations`) against `drug_interactions`.
3. Check each `drug_lab_rules` row against the patient's labs.
4. Attach evidence provenance (`source`, `source_setid`, `section_title`, `subsection_title`) to each triggered finding via `evidence_lookup`.
5. Sort findings by severity (High > Moderate > Low); overall risk = highest severity present, else LOW.
6. Format findings as text and build a prompt constrained to *only* those findings.
7. Call the LLM (`databricks-meta-llama-3-3-70b-instruct` via `WorkspaceClient().serving_endpoints.query(...)`) for a patient-friendly explanation — system prompt explicitly forbids inventing interactions/diagnoses and instructs deferring to a pharmacist/clinician.
8. Return `{patient_id, overall_risk, normalized_medications, drug_drug_findings, drug_lab_findings, ai_explanation}`.

**Important constraint for serving this outside a notebook**: `drug_interactions`, `lab_rules`, and `evidence_lookup` are loaded once via `spark.table(...).collect()` at notebook scope and closed over by `run_medication_safety_agent`. Any deployment target without a Spark session (e.g. a Model Serving container) needs these snapshotted into plain Python data at model-build time, not re-queried live.

## Glue code (from prose description — not yet verified against literal source)

- `create_patient(patient_id, age, medications, conditions=None, labs=None, allergies=None)` — standardizes a patient dict.
- `analyze_patient(patient)` — runs `run_medication_safety_agent`, returns `{raw_result, structured_report}`.
- `build_structured_report(result)` — reshapes the raw engine output into `{patient_id, overall_risk, summary: {drug_drug_count, drug_lab_count, total_findings}, normalized_medications, drug_drug_findings, drug_lab_findings, ai_explanation}`.
- `medication_safety_api(patient)` — the intended single application entry point: `create_patient` input in, `structured_report` out. This is the function the Model Serving endpoint wraps.

## Synthetic test patients

| Patient | Medications | Labs | Expected overall risk |
|---|---|---|---|
| P001 | Lisinopril, Spironolactone, Ibuprofen | K 5.1, eGFR 55 | HIGH |
| P002 | Lisinopril, Ibuprofen | K 4.4, eGFR 82 | MODERATE |
| P003 | Spironolactone, Ibuprofen | K 4.6, eGFR 48 | MODERATE (1 drug-drug + 1 drug-lab finding) |

P003 is the primary validation patient — confirmed end-to-end output (structured report + AI explanation) exists from a real run.

## Design principles carried forward

1. **Rules determine facts; the LLM only explains them** — the model never decides whether an interaction exists, and is explicitly instructed not to invent findings.
2. **Evidence and rules are separate** — `source_evidence` supports rules, but isn't itself executable.
3. **Prototype thresholds are flagged as such** — `threshold_basis` distinguishes "evidence supports the clinical relationship" from "evidence establishes this exact numeric cutoff."
4. **Findings are traceable**: Finding → Rule → Evidence ID → Source.
5. **Structured output precedes UI** — the backend produces a clean JSON contract before any interface is built against it.

## What's superseded by the current plan

The original notes' "Recommended Next Phase" (Windows dev environment, Databricks CLI/OAuth setup, extracting notebook logic into local modules, building a Streamlit UI) is superseded by the plan for this repo: expose the existing Databricks agent via a Model Serving endpoint and build a FastAPI + Jinja2/Alpine web client against it instead of Streamlit — see the top-level [README](../README.md) for the resulting architecture.
