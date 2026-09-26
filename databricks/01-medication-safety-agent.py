# Databricks notebook source
## Imports & Configuration

from itertools import combinations
import json
import requests

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

print("Configuration loaded.")

# COMMAND ----------

## RxNorm Normalization Functions

def get_rxcui(drug_name):
    """
    Look up a medication name in RxNorm and return its RxCUI.
    """

    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

    response = requests.get(
        url,
        params={"name": drug_name},
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    rxnorm_ids = data.get("idGroup", {}).get("rxnormId", [])

    if rxnorm_ids:
        return rxnorm_ids[0]

    return None


def normalize_drug(drug_name):
    """
    Normalize one medication using RxNorm.
    """

    rxcui = get_rxcui(drug_name)

    if not rxcui:
        return {
            "input_name": drug_name,
            "rxcui": None,
            "normalized_name": None
        }

    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    properties = data.get("properties", {})

    return {
        "input_name": drug_name,
        "rxcui": rxcui,
        "normalized_name": properties.get("name")
    }


def normalize_medication_list(medications):
    """
    Normalize all medications in a patient's medication list.
    """

    normalized_medications = []

    for drug in medications:
        normalized_medications.append(
            normalize_drug(drug)
        )

    return normalized_medications


print("RxNorm normalization functions loaded.")

# COMMAND ----------

## Load Safety Rules & Evidence from Catalog

drug_interactions = [
    row.asDict()
    for row in spark.table(
        "healthcare.medication_safety.drug_interactions"
    ).collect()
]

lab_rules = [
    row.asDict()
    for row in spark.table(
        "healthcare.medication_safety.drug_lab_rules"
    ).collect()
]

source_evidence = [
    row.asDict()
    for row in spark.table(
        "healthcare.medication_safety.source_evidence"
    ).collect()
]

evidence_lookup = {
    row["evidence_id"]: row
    for row in source_evidence
}

print("Drug interaction rules loaded:", len(drug_interactions))
print("Drug-lab rules loaded:", len(lab_rules))
print("Source evidence records loaded:", len(source_evidence))

# COMMAND ----------

## Medication Safety Agent

def run_medication_safety_agent(patient):

    # ---------------------------------------------------------
    # 1. Normalize Patient Medications with RxNorm
    # ---------------------------------------------------------

    normalized_medications = normalize_medication_list(
        patient["medications"]
    )

    patient_rxcuis = [
        medication["rxcui"]
        for medication in normalized_medications
        if medication["rxcui"] is not None
    ]


    # ---------------------------------------------------------
    # 2. Check Drug-Drug Interactions Using RxCUI
    # ---------------------------------------------------------

    found_interactions = []

    for rxcui1, rxcui2 in combinations(patient_rxcuis, 2):

        for interaction in drug_interactions:

            rule_a = interaction["drug_a_rxcui"]
            rule_b = interaction["drug_b_rxcui"]

            match = (
                rxcui1 == rule_a
                and rxcui2 == rule_b
            ) or (
                rxcui2 == rule_a
                and rxcui1 == rule_b
            )

            if match:

                finding = interaction.copy()

                # Attach evidence provenance
                evidence_id = interaction.get("evidence_id")

                if evidence_id:

                    evidence = evidence_lookup.get(evidence_id)

                    if evidence:

                        finding["evidence"] = {
                            "evidence_id": evidence["evidence_id"],
                            "source": evidence["source"],
                            "source_setid": evidence["source_setid"],
                            "section_title": evidence["section_title"],
                            "subsection_title": evidence["subsection_title"]
                        }

                found_interactions.append(finding)


    # ---------------------------------------------------------
    # 3. Check Drug-Lab Rules Using RxCUI
    # ---------------------------------------------------------

    drug_lab_findings = []

    for rule in lab_rules:

        drug_present = rule["drug_rxcui"] in patient_rxcuis

        lab_name = rule["lab"]
        lab_value = patient["labs"].get(lab_name)

        if drug_present and lab_value is not None:

            triggered = False

            if (
                rule["operator"] == ">"
                and lab_value > rule["threshold"]
            ):
                triggered = True

            elif (
                rule["operator"] == "<"
                and lab_value < rule["threshold"]
            ):
                triggered = True

            if triggered:

                finding = {
                    "drug": rule["drug"],
                    "drug_rxcui": rule["drug_rxcui"],
                    "lab": lab_name,
                    "value": lab_value,
                    "operator": rule["operator"],
                    "threshold": rule["threshold"],
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "threshold_basis": rule.get("threshold_basis")
                }

                # Attach Drug-Lab evidence provenance
                evidence_id = rule.get("evidence_id")

                if evidence_id:

                    evidence = evidence_lookup.get(evidence_id)

                    if evidence:

                        finding["evidence"] = {
                            "evidence_id": evidence["evidence_id"],
                            "source": evidence["source"],
                            "source_setid": evidence["source_setid"],
                            "section_title": evidence["section_title"],
                            "subsection_title": evidence["subsection_title"]
                        }

                drug_lab_findings.append(finding)
                
    # ---------------------------------------------------------
    # Sort Findings by Severity
    # ---------------------------------------------------------

    severity_rank = {
        "High": 1,
        "Moderate": 2,
        "Low": 3
    }

    found_interactions.sort(
        key=lambda item: severity_rank.get(
            item["severity"],
            99
        )
    )

    drug_lab_findings.sort(
        key=lambda item: severity_rank.get(
            item["severity"],
            99
        )
    )

    # ---------------------------------------------------------
    # 4. Calculate Overall Risk
    # ---------------------------------------------------------

    severities = []

    for item in found_interactions:
        severities.append(item["severity"])

    for item in drug_lab_findings:
        severities.append(item["severity"])

    if "High" in severities:
        overall_risk = "HIGH"

    elif "Moderate" in severities:
        overall_risk = "MODERATE"

    else:
        overall_risk = "LOW"


    # ---------------------------------------------------------
    # 5. Format Drug-Drug Findings for the LLM
    # ---------------------------------------------------------

    drug_drug_text = ""

    for item in found_interactions:

        drug_drug_text += (
            f'- {item["severity"]}: '
            f'{item["drug_a"]} + {item["drug_b"]} - '
            f'{item["interaction"]}\n'
        )


    # ---------------------------------------------------------
    # 6. Format Drug-Lab Findings for the LLM
    # ---------------------------------------------------------

    drug_lab_text = ""

    for item in drug_lab_findings:

        drug_lab_text += (
            f'- {item["severity"]}: '
            f'{item["drug"]} - '
            f'{item["lab"]} = {item["value"]} - '
            f'{item["message"]}\n'
        )


    # ---------------------------------------------------------
    # 7. Build Grounded LLM Prompt
    # ---------------------------------------------------------

    patient_prompt = f"""
Patient ID: {patient["patient_id"]}
Age: {patient["age"]}

Overall Safety Risk: {overall_risk}

DRUG-DRUG FINDINGS
{drug_drug_text if drug_drug_text else "No findings."}

DRUG-LAB FINDINGS
{drug_lab_text if drug_lab_text else "No findings."}

Using only the findings above, explain the most important
medication safety concerns for this fictional patient.

Prioritize high-severity findings.
Do not invent additional interactions or diagnoses.
"""


    # ---------------------------------------------------------
    # 8. Generate AI Explanation
    # ---------------------------------------------------------

    w = WorkspaceClient()

    response = w.serving_endpoints.query(
        name=MODEL_ENDPOINT,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="""
You are a medication safety explanation assistant.

Use only the medication safety findings supplied by the safety engine.
Do not invent additional drug interactions, diagnoses, or lab abnormalities.
Prioritize higher-severity findings.
Explain the findings in clear, patient-friendly language.
Do not provide a diagnosis.
When appropriate, recommend discussion with a pharmacist or healthcare professional.
"""
            ),
            ChatMessage(
                role=ChatMessageRole.USER,
                content=patient_prompt
            )
        ],
        max_tokens=500
    )

    ai_explanation = response.choices[0].message.content


    # ---------------------------------------------------------
    # 9. Return Complete Agent Result
    # ---------------------------------------------------------

    return {
        "patient_id": patient["patient_id"],
        "overall_risk": overall_risk,
        "normalized_medications": normalized_medications,
        "drug_drug_findings": found_interactions,
        "drug_lab_findings": drug_lab_findings,
        "ai_explanation": ai_explanation
    }


print(
    "RxNorm + Drug-Drug + Drug-Lab provenance "
    "Medication Safety Agent loaded."
)

# COMMAND ----------

## Patient Input Helper

def create_patient(
    patient_id,
    age,
    medications,
    conditions=None,
    labs=None,
    allergies=None
):
    """
    Create a standardized patient record for the
    Medication Safety Agent.

    Synthetic data only for this prototype.
    """

    if conditions is None:
        conditions = []

    if labs is None:
        labs = {}

    if allergies is None:
        allergies = []

    patient = {
        "patient_id": patient_id,
        "age": age,
        "medications": medications,
        "conditions": conditions,
        "labs": labs,
        "allergies": allergies
    }

    return patient


print("Patient input helper loaded.")

# COMMAND ----------

## Synthetic Patient P001

patient_p001 = create_patient(
    patient_id="P001",

    age=67,

    medications=[
        "Lisinopril",
        "Spironolactone",
        "Ibuprofen"
    ],

    conditions=[
        "Hypertension"
    ],

    labs={
        "potassium": 5.1,
        "egfr": 55
    },

    allergies=[
        "Penicillin"
    ]
)

print(patient_p001)

# COMMAND ----------

## Synthetic Patient P002

patient_p002 = create_patient(
    patient_id="P002",

    age=52,

    medications=[
        "Lisinopril",
        "Ibuprofen"
    ],

    conditions=[
        "Hypertension"
    ],

    labs={
        "potassium": 4.4,
        "egfr": 82
    },

    allergies=[]
)

print(patient_p002)

# COMMAND ----------

## Synthetic Patient P003

patient_p003 = create_patient(
    patient_id="P003",

    age=61,

    medications=[
        "Spironolactone",
        "Ibuprofen"
    ],

    conditions=[
        "Hypertension"
    ],

    labs={
        "potassium": 4.6,
        "egfr": 48
    },

    allergies=[]
)

print(patient_p003)

# COMMAND ----------

## Structured Report Builder

def build_structured_report(result):
    """
    Convert the raw agent result into a clean,
    machine-readable report structure.
    """

    report = {
        "patient_id": result["patient_id"],

        "overall_risk": result["overall_risk"],

        "summary": {
            "drug_drug_count":
                len(result["drug_drug_findings"]),

            "drug_lab_count":
                len(result["drug_lab_findings"]),

            "total_findings":
                len(result["drug_drug_findings"])
                + len(result["drug_lab_findings"])
        },

        "normalized_medications": [],

        "drug_drug_findings": [],

        "drug_lab_findings": [],

        "ai_explanation":
            result["ai_explanation"]
    }


    # ---------------------------------------------------------
    # Normalized Medications
    # ---------------------------------------------------------

    for medication in result[
        "normalized_medications"
    ]:

        report[
            "normalized_medications"
        ].append({
            "input_name":
                medication["input_name"],

            "normalized_name":
                medication["normalized_name"],

            "rxcui":
                medication["rxcui"]
        })


    # ---------------------------------------------------------
    # Drug-Drug Findings
    # ---------------------------------------------------------

    for finding in result[
        "drug_drug_findings"
    ]:

        finding_report = {
            "finding_type":
                "Drug-Drug",

            "severity":
                finding["severity"],

            "drug_a":
                finding["drug_a"],

            "drug_b":
                finding["drug_b"],

            "drug_a_rxcui":
                finding.get("drug_a_rxcui"),

            "drug_b_rxcui":
                finding.get("drug_b_rxcui"),

            "concern":
                finding["interaction"],

            "evidence":
                finding.get("evidence")
        }

        report[
            "drug_drug_findings"
        ].append(
            finding_report
        )


    # ---------------------------------------------------------
    # Drug-Lab Findings
    # ---------------------------------------------------------

    for finding in result[
        "drug_lab_findings"
    ]:

        finding_report = {
            "finding_type":
                "Drug-Lab",

            "severity":
                finding["severity"],

            "drug":
                finding["drug"],

            "drug_rxcui":
                finding["drug_rxcui"],

            "lab":
                finding["lab"],

            "observed_value":
                float(finding["value"]),

            "operator":
                finding["operator"],

            "threshold":
                float(finding["threshold"]),

            "concern":
                finding["message"],

            "threshold_basis":
                finding.get(
                    "threshold_basis"
                ),

            "evidence":
                finding.get("evidence")
        }

        report[
            "drug_lab_findings"
        ].append(
            finding_report
        )


    return report


print(
    "Structured report builder loaded."
)

# COMMAND ----------

## Analysis Entry Point

def analyze_patient(patient):
    """
    Run the complete Medication Safety Agent workflow
    and return both raw and structured results.

    Synthetic data only for this prototype.
    """

    print(
        f'Analyzing patient: {patient["patient_id"]}'
    )

    raw_result = run_medication_safety_agent(
        patient
    )

    structured_report = build_structured_report(
        raw_result
    )

    print(
        f'Analysis complete. Overall risk: '
        f'{structured_report["overall_risk"]}'
    )

    return {
        "raw_result": raw_result,
        "structured_report": structured_report
    }


print(
    "Medication safety analysis entry point loaded."
)

# COMMAND ----------

## Report Formatter

def print_medication_safety_report(result):
    """
    Print a structured medication safety report.
    """

    print("\n" + "=" * 70)
    print("MEDICATION SAFETY INTELLIGENCE REPORT")
    print("=" * 70)

    print(
        "\nPatient ID:",
        result["patient_id"]
    )

    print(
        "Overall Risk:",
        result["overall_risk"]
    )

    print(
        "Drug-Drug Findings:",
        len(result["drug_drug_findings"])
    )

    print(
        "Drug-Lab Findings:",
        len(result["drug_lab_findings"])
    )


    # ---------------------------------------------------------
    # Normalized Medications
    # ---------------------------------------------------------

    print("\nNORMALIZED MEDICATIONS")
    print("=" * 70)

    for medication in result[
        "normalized_medications"
    ]:

        print(
            f'{medication["input_name"]}'
            f' → '
            f'{medication["normalized_name"]}'
            f' '
            f'(RxCUI: {medication["rxcui"]})'
        )


    # ---------------------------------------------------------
    # Drug-Drug Findings
    # ---------------------------------------------------------

    print("\nDRUG-DRUG FINDINGS")
    print("=" * 70)

    if not result["drug_drug_findings"]:
        print("No Drug-Drug findings.")

    for finding in result[
        "drug_drug_findings"
    ]:

        print(
            f'\n[{finding["severity"].upper()}] '
            f'{finding["drug_a"]} + '
            f'{finding["drug_b"]}'
        )

        print(
            "Concern:",
            finding["interaction"]
        )

        evidence = finding.get(
            "evidence"
        )

        if evidence:

            print(
                "Evidence Source:",
                evidence["source"]
            )

            print(
                "Evidence ID:",
                evidence["evidence_id"]
            )

            print(
                "DailyMed SET ID:",
                evidence["source_setid"]
            )

            print(
                "Label Section:",
                evidence["subsection_title"]
            )

        else:

            print(
                "Evidence Source:"
                " Not yet linked"
            )


    # ---------------------------------------------------------
    # Drug-Lab Findings
    # ---------------------------------------------------------

    print("\nDRUG-LAB FINDINGS")
    print("=" * 70)

    if not result["drug_lab_findings"]:
        print("No Drug-Lab findings.")

    for finding in result[
        "drug_lab_findings"
    ]:

        print(
            f'\n[{finding["severity"].upper()}] '
            f'{finding["drug"]}'
        )

        print(
            "Observed Lab:",
            f'{finding["lab"]} = '
            f'{finding["value"]}'
        )

        print(
            "Triggered Rule:",
            f'{finding["lab"]} '
            f'{finding["operator"]} '
            f'{finding["threshold"]}'
        )

        print(
            "Concern:",
            finding["message"]
        )

        evidence = finding.get(
            "evidence"
        )

        if evidence:

            print(
                "Evidence Source:",
                evidence["source"]
            )

            print(
                "Evidence ID:",
                evidence["evidence_id"]
            )

            print(
                "DailyMed SET ID:",
                evidence["source_setid"]
            )

            print(
                "Label Section:",
                evidence["subsection_title"]
            )

        else:

            print(
                "Evidence Source:"
                " Not yet linked"
            )

        threshold_basis = finding.get(
            "threshold_basis"
        )

        if threshold_basis:

            print(
                "Threshold Basis:",
                threshold_basis
            )


    # ---------------------------------------------------------
    # AI Explanation
    # ---------------------------------------------------------

    print("\nAI EXPLANATION")
    print("=" * 70)

    print(
        result["ai_explanation"]
    )

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)


print(
    "Medication safety report formatter loaded."
)

# COMMAND ----------

## Select Patient & Run Agent

selected_patient = patient_p003

analysis = analyze_patient(
    selected_patient
)

result = analysis[
    "raw_result"
]

structured_report = analysis[
    "structured_report"
]

print_medication_safety_report(
    result
)

# COMMAND ----------

## Validate Structured Report

import json

print(
    json.dumps(
        structured_report,
        indent=2,
        default=str
    )
)

# COMMAND ----------

## Application Interface

def medication_safety_api(patient):
    """
    Application-facing interface for the
    Medication Safety Intelligence Agent.

    Accepts one standardized patient dictionary
    and returns only the structured report.

    Synthetic data only for this prototype.
    """

    analysis = analyze_patient(
        patient
    )

    return analysis[
        "structured_report"
    ]


print(
    "Medication Safety application interface loaded."
)


# COMMAND ----------

## Application Interface Test

import json

api_response = medication_safety_api(
    patient_p003
)

print(
    json.dumps(
        api_response,
        indent=2,
        default=str
    )
)

# COMMAND ----------

def run_medication_safety_agent_for_serving(patient, drug_interactions, lab_rules, evidence_lookup):
    from itertools import combinations

    normalized_medications = normalize_medication_list(patient["medications"])

    patient_rxcuis = [
        medication["rxcui"]
        for medication in normalized_medications
        if medication["rxcui"] is not None
    ]

    found_interactions = []
    for rxcui1, rxcui2 in combinations(patient_rxcuis, 2):
        for interaction in drug_interactions:
            rule_a = interaction["drug_a_rxcui"]
            rule_b = interaction["drug_b_rxcui"]
            match = (rxcui1 == rule_a and rxcui2 == rule_b) or (rxcui2 == rule_a and rxcui1 == rule_b)
            if match:
                finding = interaction.copy()
                evidence_id = interaction.get("evidence_id")
                if evidence_id:
                    evidence = evidence_lookup.get(evidence_id)
                    if evidence:
                        finding["evidence"] = {
                            "evidence_id": evidence["evidence_id"],
                            "source": evidence["source"],
                            "source_setid": evidence["source_setid"],
                            "section_title": evidence["section_title"],
                            "subsection_title": evidence["subsection_title"],
                        }
                found_interactions.append(finding)

    drug_lab_findings = []
    for rule in lab_rules:
        drug_present = rule["drug_rxcui"] in patient_rxcuis
        lab_name = rule["lab"]
        lab_value = patient["labs"].get(lab_name)
        if drug_present and lab_value is not None:
            triggered = False
            if rule["operator"] == ">" and lab_value > rule["threshold"]:
                triggered = True
            elif rule["operator"] == "<" and lab_value < rule["threshold"]:
                triggered = True
            if triggered:
                finding = {
                    "drug": rule["drug"],
                    "drug_rxcui": rule["drug_rxcui"],
                    "lab": lab_name,
                    "value": lab_value,
                    "operator": rule["operator"],
                    "threshold": rule["threshold"],
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "threshold_basis": rule.get("threshold_basis"),
                }
                evidence_id = rule.get("evidence_id")
                if evidence_id:
                    evidence = evidence_lookup.get(evidence_id)
                    if evidence:
                        finding["evidence"] = {
                            "evidence_id": evidence["evidence_id"],
                            "source": evidence["source"],
                            "source_setid": evidence["source_setid"],
                            "section_title": evidence["section_title"],
                            "subsection_title": evidence["subsection_title"],
                        }
                drug_lab_findings.append(finding)

    severity_rank = {"High": 1, "Moderate": 2, "Low": 3}
    found_interactions.sort(key=lambda item: severity_rank.get(item["severity"], 99))
    drug_lab_findings.sort(key=lambda item: severity_rank.get(item["severity"], 99))

    severities = [item["severity"] for item in found_interactions] + [item["severity"] for item in drug_lab_findings]
    if "High" in severities:
        overall_risk = "HIGH"
    elif "Moderate" in severities:
        overall_risk = "MODERATE"
    else:
        overall_risk = "LOW"

    drug_drug_text = "".join(
        f'- {item["severity"]}: {item["drug_a"]} + {item["drug_b"]} - {item["interaction"]}\n'
        for item in found_interactions
    )
    drug_lab_text = "".join(
        f'- {item["severity"]}: {item["drug"]} - {item["lab"]} = {item["value"]} - {item["message"]}\n'
        for item in drug_lab_findings
    )

    patient_prompt = f"""
Patient ID: {patient["patient_id"]}
Age: {patient["age"]}

Overall Safety Risk: {overall_risk}

DRUG-DRUG FINDINGS
{drug_drug_text if drug_drug_text else "No findings."}

DRUG-LAB FINDINGS
{drug_lab_text if drug_lab_text else "No findings."}

Using only the findings above, explain the most important
medication safety concerns for this fictional patient.

Prioritize high-severity findings.
Do not invent additional interactions or diagnoses.
"""

    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=MODEL_ENDPOINT,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="""
You are a medication safety explanation assistant.

Use only the medication safety findings supplied by the safety engine.
Do not invent additional drug interactions, diagnoses, or lab abnormalities.
Prioritize higher-severity findings.
Explain the findings in clear, patient-friendly language.
Do not provide a diagnosis.
When appropriate, recommend discussion with a pharmacist or healthcare professional.
""",
            ),
            ChatMessage(role=ChatMessageRole.USER, content=patient_prompt),
        ],
        max_tokens=500,
    )
    ai_explanation = response.choices[0].message.content
    # Real runtime identifier from the LLM call's own response (e.g.
    # "meta-llama-3.3-70b-instruct-121024") -- more precise than just
    # echoing the endpoint name we configured, and reflects whatever
    # Databricks actually served this request, not a hardcoded guess.
    explanation_model = getattr(response, "model", None) or MODEL_ENDPOINT

    return {
        "patient_id": patient["patient_id"],
        "overall_risk": overall_risk,
        "normalized_medications": normalized_medications,
        "drug_drug_findings": found_interactions,
        "drug_lab_findings": drug_lab_findings,
        "ai_explanation": ai_explanation,
        "explanation_provider": "Databricks",
        "explanation_model": explanation_model,
    }


def medication_safety_api_for_serving(patient, drug_interactions, lab_rules, evidence_lookup):
    result = run_medication_safety_agent_for_serving(patient, drug_interactions, lab_rules, evidence_lookup)
    return {
        "patient_id": result["patient_id"],
        "overall_risk": result["overall_risk"],
        "summary": {
            "drug_drug_count": len(result["drug_drug_findings"]),
            "drug_lab_count": len(result["drug_lab_findings"]),
            "total_findings": len(result["drug_drug_findings"]) + len(result["drug_lab_findings"]),
        },
        "normalized_medications": result["normalized_medications"],
        "drug_drug_findings": result["drug_drug_findings"],
        "drug_lab_findings": result["drug_lab_findings"],
        "ai_explanation": result["ai_explanation"],
        "explanation_provider": result["explanation_provider"],
        "explanation_model": result["explanation_model"],
    }

print("Serving-ready (parameterized) functions loaded.")


# COMMAND ----------

test_result = medication_safety_api_for_serving(patient_p003, drug_interactions, lab_rules, evidence_lookup)
print("Overall risk:", test_result["overall_risk"])
print("Total findings:", test_result["summary"]["total_findings"])
# Expect: MODERATE, 2 -- matching your original medication_safety_api(patient_p003) result


# COMMAND ----------

## Pre-Add Safety Check (hybrid architecture)
#
# Added so the FastAPI app's Medication Management tab can get a clinical
# pre-add safety check without medication-lookup-api. Stateless by design,
# matching this model's existing pattern: the caller (FastAPI, which now
# owns patient_medications CRUD) passes the patient's current medications
# directly — this model has no SQL access and none is added for this
# feature; it reuses the SAME baked-in drug_interactions/lab_rules/
# evidence_lookup snapshot the main safety analysis already uses.
#
# Falls back to a live openFDA label lookup only when the static rule
# table has nothing for a pair. The verbatim excerpt is the fact; the LLM
# (the same shared foundation-model endpoint already used for the main
# explanation) only paraphrases it — it never decides whether an
# interaction exists. See docs/databricks-endpoint-lifecycle.md.

_OPENFDA_SECTION_LOINC = {
    "drug_interactions": "34073-7",
    "warnings_and_cautions": "43685-7",
    "contraindications": "34070-3",
}


def _openfda_label(generic_name):
    generic_name = (generic_name or "").strip()
    if not generic_name:
        return None
    try:
        response = requests.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'openfda.generic_name:"{generic_name.upper()}"', "limit": 1},
            timeout=10,
        )
        if response.status_code != 200:
            return None
        results = response.json().get("results") or []
        return results[0] if results else None
    except Exception:
        # openFDA is a best-effort fallback signal — a lookup failure must
        # not break pre_add_check, it just means no dynamic signal here.
        return None


def _openfda_mention(label, other_name):
    other_name = (other_name or "").strip().lower()
    if not other_name or not label:
        return None
    for field in ("drug_interactions", "warnings_and_cautions", "contraindications"):
        values = label.get(field)
        if not values:
            continue
        text = " ".join(values) if isinstance(values, list) else str(values)
        idx = text.lower().find(other_name)
        if idx == -1:
            continue
        start = max(0, idx - 200)
        end = min(len(text), idx + len(other_name) + 400)
        return {
            "field": field,
            "section_code": _OPENFDA_SECTION_LOINC.get(field, ""),
            "excerpt": text[start:end].strip(),
            "set_id": label.get("set_id"),
        }
    return None


def _openfda_signal(candidate_name, candidate_rxcui, existing_name, existing_rxcui):
    candidate_label = _openfda_label(candidate_name)
    mention = _openfda_mention(candidate_label, existing_name) if candidate_label else None
    if mention:
        return {
            **mention,
            "labeled_drug": candidate_name,
            "labeled_drug_rxcui": candidate_rxcui,
            "mentioned_drug": existing_name,
            "mentioned_drug_rxcui": existing_rxcui,
        }

    existing_label = _openfda_label(existing_name)
    mention = _openfda_mention(existing_label, candidate_name) if existing_label else None
    if mention:
        return {
            **mention,
            "labeled_drug": existing_name,
            "labeled_drug_rxcui": existing_rxcui,
            "mentioned_drug": candidate_name,
            "mentioned_drug_rxcui": candidate_rxcui,
        }

    return None


def _llm_explain_excerpt(labeled_drug, mentioned_drug, excerpt):
    try:
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=MODEL_ENDPOINT,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content=(
                        "You explain a single excerpt from an FDA drug label in plain language for a "
                        "patient. Only explain what the excerpt itself says. Never add a claim, a "
                        "severity judgment, or a recommendation that is not directly supported by the "
                        "excerpt's own wording. Keep it to 2-3 short sentences."
                    ),
                ),
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=(
                        f"This excerpt is from the FDA label for {labeled_drug}, in the context of "
                        f"{mentioned_drug}:\n\n{excerpt}\n\n"
                        "Explain in plain language what this excerpt says."
                    ),
                ),
            ],
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # The raw excerpt (already surfaced separately) remains available
        # even if the explanation call fails.
        return None


def pre_add_check_for_serving(
    candidate_rxcui,
    candidate_name,
    candidate_ingredients,
    current_medications,
    labs,
    drug_interactions,
    lab_rules,
    evidence_lookup,
):
    labs = labs or {}
    normalized_labs = {str(key).lower(): value for key, value in labs.items()}

    new_ingredients = candidate_ingredients or []

    current_ingredients = []
    for medication in current_medications or []:
        for ingredient in medication.get("ingredients") or []:
            current_ingredients.append(
                {
                    "selected_name": medication.get("selected_name"),
                    "rxcui": str(ingredient.get("rxcui") or ""),
                    "name": ingredient.get("name"),
                }
            )

    drug_drug_findings = []

    # Static rule-table matching (same baked-in snapshot as safety analysis).
    for new_ingredient in new_ingredients:
        new_rxcui = str(new_ingredient.get("rxcui") or "")
        for current_ingredient in current_ingredients:
            current_rxcui = str(current_ingredient.get("rxcui") or "")
            for rule in drug_interactions:
                rule_a = str(rule.get("drug_a_rxcui") or "")
                rule_b = str(rule.get("drug_b_rxcui") or "")
                match = (new_rxcui == rule_a and current_rxcui == rule_b) or (
                    new_rxcui == rule_b and current_rxcui == rule_a
                )
                if not match:
                    continue
                drug_drug_findings.append(
                    {
                        "finding_type": "Drug-Drug",
                        "severity": rule.get("severity"),
                        "new_medication": candidate_name,
                        "existing_medication": current_ingredient.get("selected_name"),
                        "drug_a": rule.get("drug_a"),
                        "drug_b": rule.get("drug_b"),
                        "concern": rule.get("interaction"),
                        "evidence_id": rule.get("evidence_id"),
                        "source": rule.get("source"),
                        "review_status": rule.get("review_status"),
                    }
                )

    # Dynamic openFDA fallback — only when the static table found nothing.
    if not drug_drug_findings:
        for new_ingredient in new_ingredients:
            if drug_drug_findings:
                break
            new_rxcui_check = str(new_ingredient.get("rxcui") or "")
            for current_ingredient in current_ingredients:
                current_rxcui_check = str(current_ingredient.get("rxcui") or "")

                # Same ingredient already on file — searching a drug's own
                # label for its own name would always "match" itself.
                if new_rxcui_check and new_rxcui_check == current_rxcui_check:
                    continue

                signal = _openfda_signal(
                    candidate_name=new_ingredient.get("name"),
                    candidate_rxcui=new_rxcui_check,
                    existing_name=current_ingredient.get("name"),
                    existing_rxcui=current_rxcui_check,
                )
                if not signal:
                    continue

                explanation = _llm_explain_excerpt(
                    signal["labeled_drug"], signal["mentioned_drug"], signal["excerpt"]
                )

                drug_drug_findings.append(
                    {
                        "finding_type": "Drug-Drug",
                        "severity": "Unreviewed",
                        "new_medication": candidate_name,
                        "new_medication_rxcui": new_rxcui_check,
                        "existing_medication": current_ingredient.get("selected_name"),
                        "existing_medication_rxcui": current_rxcui_check,
                        "drug_a": new_ingredient.get("name") or candidate_name,
                        "drug_b": current_ingredient.get("name") or current_ingredient.get("selected_name"),
                        "concern": explanation or signal["excerpt"],
                        "trusted_source_excerpt": signal["excerpt"],
                        "trusted_source_field": signal["field"],
                        "trusted_source_section_code": signal["section_code"],
                        "trusted_source_set_id": signal["set_id"],
                        "labeled_drug": signal["labeled_drug"],
                        "ai_explanation": explanation,
                        "source": "openFDA",
                        "evidence_id": "FDA-" + "-".join(sorted([new_rxcui_check, current_rxcui_check])),
                        "review_status": "Auto-discovered, unreviewed",
                    }
                )
                break

    drug_lab_findings = []
    for new_ingredient in new_ingredients:
        new_rxcui = str(new_ingredient.get("rxcui") or "")
        for rule in lab_rules:
            if new_rxcui != str(rule.get("drug_rxcui") or ""):
                continue
            lab_name = str(rule.get("lab") or "").lower()
            if lab_name not in normalized_labs:
                continue
            try:
                observed = float(normalized_labs[lab_name])
                threshold = float(rule.get("threshold"))
            except (TypeError, ValueError):
                continue

            operator = rule.get("operator")
            triggered = (operator == ">" and observed > threshold) or (operator == "<" and observed < threshold)
            if not triggered:
                continue

            evidence = evidence_lookup.get(rule.get("evidence_id")) or {}
            drug_lab_findings.append(
                {
                    "finding_type": "Drug-Lab",
                    "severity": rule.get("severity"),
                    "drug": rule.get("drug"),
                    "drug_rxcui": rule.get("drug_rxcui"),
                    "lab": lab_name,
                    "observed_value": observed,
                    "operator": operator,
                    "threshold": threshold,
                    "concern": rule.get("message"),
                    "threshold_basis": rule.get("threshold_basis"),
                    "evidence_id": rule.get("evidence_id"),
                    "source": evidence.get("source"),
                }
            )

    severity_order = {"High": 1, "Moderate": 2, "Low": 3, "Unreviewed": 4}
    drug_drug_findings.sort(key=lambda item: severity_order.get(item.get("severity"), 99))
    drug_lab_findings.sort(key=lambda item: severity_order.get(item.get("severity"), 99))

    all_findings = drug_drug_findings + drug_lab_findings

    safety_review_required = any(
        f.get("severity") in {"High", "Moderate", "Unreviewed"} for f in all_findings
    )
    has_unreviewed = any(f.get("review_status") == "Auto-discovered, unreviewed" for f in all_findings)

    if has_unreviewed:
        message = (
            "A potential interaction was found in FDA drug labeling that is not yet part of the "
            "reviewed prototype rules. Review the excerpt and explanation below and confirm with the "
            "prescribing clinician or pharmacist before starting a proposed medication. Existing "
            "medications should still be recorded accurately."
        )
    elif all_findings:
        message = (
            "Safety findings were identified from the current prototype rules. Review the findings "
            "before starting a proposed medication. Existing medications should still be recorded "
            "accurately."
        )
    else:
        message = (
            "No supported safety concern was identified from the current prototype rules and "
            "available patient data. This does not prove that the medication is risk-free."
        )

    return {
        "candidate_rxcui": candidate_rxcui,
        "candidate_name": candidate_name,
        "drug_drug_findings": drug_drug_findings,
        "drug_lab_findings": drug_lab_findings,
        "total_findings": len(all_findings),
        "safety_review_required": safety_review_required,
        "message": message,
        "coverage_note": (
            "Pre-add screening uses the prototype Databricks rule tables baked into this model "
            "version, plus a live openFDA fallback when the static tables have nothing for a pair."
        ),
    }


print("Pre-add safety check function loaded.")


# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from mlflow.models.resources import DatabricksServingEndpoint


class MedicationSafetyModel(PythonModel):
    def __init__(self, drug_interactions, lab_rules, evidence_lookup):
        self.drug_interactions = drug_interactions
        self.lab_rules = lab_rules
        self.evidence_lookup = evidence_lookup

    def predict(self, context, model_input):
        records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else model_input
        results = []
        for p in records:
            if p.get("action") == "pre_add_check":
                # A missing optional column in a pandas-sourced record comes
                # back as NaN (a *truthy* float), not None -- `x or default`
                # doesn't catch it, so extract explicitly by expected type.
                context_raw = p.get("context_json")
                ctx = json.loads(context_raw) if isinstance(context_raw, str) and context_raw else {}
                labs_raw = p.get("labs")
                labs = labs_raw if isinstance(labs_raw, dict) else None
                results.append(
                    pre_add_check_for_serving(
                        candidate_rxcui=p.get("candidate_rxcui"),
                        candidate_name=ctx.get("candidate_name"),
                        candidate_ingredients=ctx.get("candidate_ingredients") or [],
                        current_medications=ctx.get("current_medications") or [],
                        labs=labs,
                        drug_interactions=self.drug_interactions,
                        lab_rules=self.lab_rules,
                        evidence_lookup=self.evidence_lookup,
                    )
                )
            else:
                # No action field (or not "pre_add_check") -> exactly today's
                # behavior: the whole record is a flat patient dict.
                results.append(
                    medication_safety_api_for_serving(p, self.drug_interactions, self.lab_rules, self.evidence_lookup)
                )
        return results if len(results) > 1 else results[0]


# COMMAND ----------

## Example rows for signature inference
#
# Three rows, each with a DIFFERENT set of populated keys, so MLflow infers
# every column that isn't common to all three as optional/nullable. This is
# what lets the analyze path (row 1, unchanged) keep working exactly as
# before while pre_add_check's new fields (rows 2-3) are genuinely optional
# on the analyze path, and vice versa. Row 3 additionally has no "labs" key
# at all, so a missing-labs pre_add_check call is accepted (never invent
# lab values when they're unavailable).

pre_add_check_example_with_labs = {
    "action": "pre_add_check",
    "candidate_rxcui": "372300",
    "labs": {"egfr": 55, "potassium": 5.1},
    "context_json": json.dumps(
        {
            "candidate_name": "gemfibrozil Oral Tablet",
            "candidate_ingredients": [{"rxcui": "4719", "name": "gemfibrozil"}],
            "current_medications": [
                {
                    "selected_name": "atorvastatin 20 MG Oral Tablet [Lipitor]",
                    "ingredients": [{"rxcui": "83367", "name": "atorvastatin"}],
                }
            ],
        }
    ),
}

pre_add_check_example_no_labs = {
    "action": "pre_add_check",
    "candidate_rxcui": "372300",
    "context_json": json.dumps(
        {
            "candidate_name": "gemfibrozil Oral Tablet",
            "candidate_ingredients": [{"rxcui": "4719", "name": "gemfibrozil"}],
            "current_medications": [],
        }
    ),
}

signature_examples = [patient_p003, pre_add_check_example_with_labs, pre_add_check_example_no_labs]

print("Signature example rows prepared:", len(signature_examples))


# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Compute the signature explicitly by actually running predict() on the
# example DataFrame, instead of relying on log_model(input_example=...) to
# infer it implicitly. That implicit path silently produces NO signature
# (not even an error) if predict() raises internally -- explicit computation
# surfaces the real exception here instead of a confusing downstream
# "Model passed for registration did not contain any signature metadata"
# error at registration time.
from mlflow.models import infer_signature

_signature_input_df = pd.DataFrame(signature_examples)
_signature_probe_model = MedicationSafetyModel(drug_interactions, lab_rules, evidence_lookup)
_signature_output = _signature_probe_model.predict(None, _signature_input_df)
model_signature = infer_signature(_signature_input_df, _signature_output)
print(model_signature)

with mlflow.start_run():
    logged_model = mlflow.pyfunc.log_model(
        artifact_path="medication_safety_agent",
        python_model=MedicationSafetyModel(drug_interactions, lab_rules, evidence_lookup),
        registered_model_name="healthcare.medication_safety.medication_safety_agent",
        pip_requirements=["requests", "databricks-sdk", "pandas", "mlflow"],
        resources=[DatabricksServingEndpoint(endpoint_name=MODEL_ENDPOINT)],
        signature=model_signature,
    )

print("Model logged and registered:", logged_model.model_uri)


# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

try:
    endpoint = w.serving_endpoints.create(
        name="medication-safety-agent",
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name="healthcare.medication_safety.medication_safety_agent",
                    entity_version="1",
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        ),
    )
    print("Endpoint creation submitted:", endpoint)
except Exception as e:
    print("Endpoint creation FAILED:")
    print(type(e).__name__, str(e))


# COMMAND ----------

w = WorkspaceClient()
response = w.serving_endpoints.query(
    name="medication-safety-agent",
    dataframe_records=[patient_p003],
)
print(response)


# COMMAND ----------

from decimal import Decimal

def convert_decimals(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [convert_decimals(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    return obj

drug_interactions_clean = convert_decimals(drug_interactions)
lab_rules_clean = convert_decimals(lab_rules)
evidence_lookup_clean = convert_decimals(evidence_lookup)

# quick re-check before redeploying
test_result = medication_safety_api_for_serving(patient_p003, drug_interactions_clean, lab_rules_clean, evidence_lookup_clean)
print(test_result["overall_risk"], test_result["summary"])


# COMMAND ----------

_signature_probe_model_clean = MedicationSafetyModel(drug_interactions_clean, lab_rules_clean, evidence_lookup_clean)
_signature_output_clean = _signature_probe_model_clean.predict(None, _signature_input_df)
model_signature_clean = infer_signature(_signature_input_df, _signature_output_clean)
print(model_signature_clean)

with mlflow.start_run():
    logged_model = mlflow.pyfunc.log_model(
        artifact_path="medication_safety_agent",
        python_model=MedicationSafetyModel(drug_interactions_clean, lab_rules_clean, evidence_lookup_clean),
        registered_model_name="healthcare.medication_safety.medication_safety_agent",
        pip_requirements=["requests", "databricks-sdk", "pandas", "mlflow"],
        resources=[DatabricksServingEndpoint(endpoint_name=MODEL_ENDPOINT)],
        signature=model_signature_clean,
    )

print("New model version logged:", logged_model.model_uri)


# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions("name='healthcare.medication_safety.medication_safety_agent'")
latest_version = max(int(v.version) for v in versions)
print("Latest version:", latest_version)

w.serving_endpoints.update_config(
    name="medication-safety-agent",
    served_entities=[
        ServedEntityInput(
            entity_name="healthcare.medication_safety.medication_safety_agent",
            entity_version=str(latest_version),
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ],
)
print("Endpoint update submitted — check Serving tab for it to go Ready again.")
