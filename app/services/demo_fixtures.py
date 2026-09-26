"""Offline/demo-mode fixtures for the two Databricks-backed calls.

Used only when `settings.demo_mode` is true (see app/config.py). A fresh
clone with no Databricks workspace, profile, or credentials can then still
demonstrate the UI end to end.

These are NOT a reimplementation of the clinical rule engine. The three
seed-patient analyses below are real responses captured from a live run of
the deployed medication-safety-agent endpoint (2026-09-24), with only the AI
explanation text/provider/model replaced by an explicit demo-mode label —
the findings, evidence, and risk levels are byte-accurate to what the live
endpoint returned. The small rule table used by demo_pre_add_check() below
is transcribed from that same live capture (the model's rule table covers
exactly these three drugs: Lisinopril, Spironolactone, Ibuprofen). Outside
of these seed patients and these three drugs, demo mode has no coverage and
says so explicitly rather than fabricating a plausible-looking result.
"""

import copy
import json

_DEMO_ANALYSIS_JSON = r'''
{
  "P001": {
    "patient_id": "P001",
    "overall_risk": "HIGH",
    "summary": {"drug_drug_count": 3, "drug_lab_count": 3, "total_findings": 6},
    "normalized_medications": [
      {"input_name": "Lisinopril", "rxcui": "29046", "normalized_name": "lisinopril"},
      {"input_name": "Spironolactone", "rxcui": "9997", "normalized_name": "spironolactone"},
      {"input_name": "Ibuprofen", "rxcui": "5640", "normalized_name": "ibuprofen"}
    ],
    "drug_drug_findings": [
      {"drug_a": "Lisinopril", "drug_b": "Spironolactone", "interaction": "May increase the risk of high potassium levels.", "severity": "High", "drug_a_rxcui": "29046", "drug_b_rxcui": "9997", "evidence_id": "DM-LISINOPRIL-SPIRONOLACTONE-001", "source": "DailyMed", "review_status": "Prototype Reviewed", "evidence": {"evidence_id": "DM-LISINOPRIL-SPIRONOLACTONE-001", "source": "DailyMed", "source_setid": "34bb4602-ccff-4939-91b9-26a63bfd40f7", "section_title": "DRUG INTERACTIONS", "subsection_title": "7.1 Diuretics"}},
      {"drug_a": "Lisinopril", "drug_b": "Ibuprofen", "interaction": "May reduce blood pressure control and affect kidney function.", "severity": "Moderate", "drug_a_rxcui": "29046", "drug_b_rxcui": "5640", "evidence_id": "DM-LISINOPRIL-NSAID-001", "source": "DailyMed", "review_status": "Prototype Reviewed", "evidence": {"evidence_id": "DM-LISINOPRIL-NSAID-001", "source": "DailyMed", "source_setid": "34bb4602-ccff-4939-91b9-26a63bfd40f7", "section_title": "DRUG INTERACTIONS", "subsection_title": "7.3 Non-Steroidal Anti-Inflammatory Agents Including Selective Cyclooxygenase-2 Inhibitors (COX-2 Inhibitors)"}},
      {"drug_a": "Spironolactone", "drug_b": "Ibuprofen", "interaction": "Concomitant NSAID use may contribute to worsening renal function.", "severity": "Moderate", "drug_a_rxcui": "9997", "drug_b_rxcui": "5640", "evidence_id": "DM-SPIRONOLACTONE-NSAID-001", "source": "DailyMed", "review_status": "Prototype Reviewed", "evidence": {"evidence_id": "DM-SPIRONOLACTONE-NSAID-001", "source": "DailyMed", "source_setid": "4290b3ec-8993-47d4-bfe3-fcb81d7baed8", "section_title": "WARNINGS AND PRECAUTIONS", "subsection_title": "5.2 Hypotension and Worsening Renal Function"}}
    ],
    "drug_lab_findings": [
      {"drug": "Lisinopril", "drug_rxcui": "29046", "lab": "potassium", "value": 5.1, "operator": ">", "threshold": 5.0, "severity": "High", "message": "Elevated potassium may increase the safety concern with lisinopril.", "threshold_basis": "Prototype threshold. DailyMed supports hyperkalemia risk and potassium monitoring but does not establish the >5.0 cutoff used by this rule.", "evidence": {"evidence_id": "DM-LISINOPRIL-HYPERKALEMIA-001", "source": "DailyMed", "source_setid": "34bb4602-ccff-4939-91b9-26a63bfd40f7", "section_title": "WARNINGS AND PRECAUTIONS", "subsection_title": "5.5 Hyperkalemia"}},
      {"drug": "Spironolactone", "drug_rxcui": "9997", "lab": "potassium", "value": 5.1, "operator": ">", "threshold": 5.0, "severity": "High", "message": "Elevated potassium may increase the safety concern with spironolactone.", "threshold_basis": "Prototype threshold. DailyMed supports hyperkalemia risk and potassium monitoring but does not establish the >5.0 cutoff used by this rule.", "evidence": {"evidence_id": "DM-SPIRONOLACTONE-HYPERKALEMIA-001", "source": "DailyMed", "source_setid": "4290b3ec-8993-47d4-bfe3-fcb81d7baed8", "section_title": "WARNINGS AND PRECAUTIONS", "subsection_title": "5.1 Hyperkalemia"}},
      {"drug": "Ibuprofen", "drug_rxcui": "5640", "lab": "egfr", "value": 55, "operator": "<", "threshold": 60.0, "severity": "Moderate", "message": "Reduced kidney function may increase concern with ibuprofen.", "threshold_basis": "Prototype threshold. DailyMed supports renal risk with ibuprofen/NSAID use but does not establish the eGFR <60 cutoff used by this rule.", "evidence": {"evidence_id": "DM-IBUPROFEN-RENAL-001", "source": "DailyMed", "source_setid": "40071818-5bf0-095b-e063-6294a90ad58a", "section_title": "WARNINGS", "subsection_title": "Renal Effects / Advanced Renal Disease"}}
    ]
  },
  "P002": {
    "patient_id": "P002",
    "overall_risk": "MODERATE",
    "summary": {"drug_drug_count": 1, "drug_lab_count": 0, "total_findings": 1},
    "normalized_medications": [
      {"input_name": "Lisinopril", "rxcui": "29046", "normalized_name": "lisinopril"},
      {"input_name": "Ibuprofen", "rxcui": "5640", "normalized_name": "ibuprofen"}
    ],
    "drug_drug_findings": [
      {"drug_a": "Lisinopril", "drug_b": "Ibuprofen", "interaction": "May reduce blood pressure control and affect kidney function.", "severity": "Moderate", "drug_a_rxcui": "29046", "drug_b_rxcui": "5640", "evidence_id": "DM-LISINOPRIL-NSAID-001", "source": "DailyMed", "review_status": "Prototype Reviewed", "evidence": {"evidence_id": "DM-LISINOPRIL-NSAID-001", "source": "DailyMed", "source_setid": "34bb4602-ccff-4939-91b9-26a63bfd40f7", "section_title": "DRUG INTERACTIONS", "subsection_title": "7.3 Non-Steroidal Anti-Inflammatory Agents Including Selective Cyclooxygenase-2 Inhibitors (COX-2 Inhibitors)"}}
    ],
    "drug_lab_findings": []
  },
  "P003": {
    "patient_id": "P003",
    "overall_risk": "MODERATE",
    "summary": {"drug_drug_count": 1, "drug_lab_count": 1, "total_findings": 2},
    "normalized_medications": [
      {"input_name": "Spironolactone", "rxcui": "9997", "normalized_name": "spironolactone"},
      {"input_name": "Ibuprofen", "rxcui": "5640", "normalized_name": "ibuprofen"}
    ],
    "drug_drug_findings": [
      {"drug_a": "Spironolactone", "drug_b": "Ibuprofen", "interaction": "Concomitant NSAID use may contribute to worsening renal function.", "severity": "Moderate", "drug_a_rxcui": "9997", "drug_b_rxcui": "5640", "evidence_id": "DM-SPIRONOLACTONE-NSAID-001", "source": "DailyMed", "review_status": "Prototype Reviewed", "evidence": {"evidence_id": "DM-SPIRONOLACTONE-NSAID-001", "source": "DailyMed", "source_setid": "4290b3ec-8993-47d4-bfe3-fcb81d7baed8", "section_title": "WARNINGS AND PRECAUTIONS", "subsection_title": "5.2 Hypotension and Worsening Renal Function"}}
    ],
    "drug_lab_findings": [
      {"drug": "Ibuprofen", "drug_rxcui": "5640", "lab": "egfr", "value": 48, "operator": "<", "threshold": 60.0, "severity": "Moderate", "message": "Reduced kidney function may increase concern with ibuprofen.", "threshold_basis": "Prototype threshold. DailyMed supports renal risk with ibuprofen/NSAID use but does not establish the eGFR <60 cutoff used by this rule.", "evidence": {"evidence_id": "DM-IBUPROFEN-RENAL-001", "source": "DailyMed", "source_setid": "40071818-5bf0-095b-e063-6294a90ad58a", "section_title": "WARNINGS", "subsection_title": "Renal Effects / Advanced Renal Disease"}}
    ]
  }
}
'''

DEMO_ANALYSIS: dict = json.loads(_DEMO_ANALYSIS_JSON)

_DEMO_NOTE = (
    "[DEMO MODE] This is a canned, offline explanation shown because DEMO_MODE is "
    "enabled — it was not generated by a live model. The findings above are real "
    "facts captured from a live run of the deployed medication-safety-agent endpoint "
    "and replayed verbatim for demonstration purposes. "
)

for _rec in DEMO_ANALYSIS.values():
    _rec["explanation_provider"] = "Demo (offline, canned fixture)"
    _rec["explanation_model"] = "none — DEMO_MODE replay, no live model called"

_DEMO_ANALYSIS_EXPLANATIONS = {
    "P001": (
        "This patient has several medication safety concerns that need attention. The most "
        "important concerns are related to high potassium levels. The patient is taking "
        "Lisinopril and Spironolactone together, which may increase the risk of high potassium "
        "levels. Additionally, the patient's lab results show an elevated potassium level of "
        "5.1, which increases the safety concerns with both Lisinopril and Spironolactone. It's "
        "recommended that the patient discusses these findings with their pharmacist or "
        "healthcare professional to understand the potential risks and determine the best "
        "course of action to ensure their safety while taking these medications."
    ),
    "P002": (
        "For patient P002, the most important medication safety concern is related to the "
        "combination of Lisinopril and Ibuprofen. Taking these two medications together may "
        "reduce the effectiveness of blood pressure control and may also affect kidney "
        "function. It's recommended that you discuss this interaction with your pharmacist or "
        "healthcare professional to understand the best course of action for your specific "
        "situation."
    ),
    "P003": (
        "The safety engine has identified two moderate-severity findings related to your "
        "medications. The combination of Spironolactone and Ibuprofen may be a concern because "
        "Ibuprofen, a non-steroidal anti-inflammatory drug (NSAID), can potentially worsen "
        "kidney function when used with other medications like Spironolactone. Additionally, "
        "the laboratory finding of reduced kidney function (eGFR = 48) is a concern when taking "
        "Ibuprofen. Discuss these findings with your pharmacist or healthcare professional to "
        "determine the best course of action for your individual situation."
    ),
}
for _pid, _text in _DEMO_ANALYSIS_EXPLANATIONS.items():
    DEMO_ANALYSIS[_pid]["ai_explanation"] = _DEMO_NOTE + _text


def demo_analyze(patient: dict) -> dict:
    """Demo-mode replacement for databricks_client.analyze_patient()."""
    patient_id = patient.get("patient_id")
    fixture = DEMO_ANALYSIS.get(patient_id)
    if fixture is not None:
        return copy.deepcopy(fixture)

    return {
        "patient_id": patient_id,
        "overall_risk": "LOW",
        "summary": {"drug_drug_count": 0, "drug_lab_count": 0, "total_findings": 0},
        "normalized_medications": [],
        "drug_drug_findings": [],
        "drug_lab_findings": [],
        "ai_explanation": (
            _DEMO_NOTE
            + f"DEMO_MODE has no canned analysis for patient '{patient_id}'. Canned coverage "
            "is limited to the three seed patients (P001, P002, P003). Disable DEMO_MODE and "
            "configure Databricks credentials to analyze other patients."
        ),
        "explanation_provider": "Demo (offline, canned fixture)",
        "explanation_model": "none — DEMO_MODE replay, no live model called",
    }


# Transcribed from the same live capture as DEMO_ANALYSIS above. This is the
# model's complete drug-drug/drug-lab rule table for its only three covered
# drugs — not a separate or expanded rule set.
_RULE_DRUG_DRUG = [
    {
        "rxcuis": {"29046", "9997"},
        "drug_a": "Lisinopril", "drug_b": "Spironolactone",
        "concern": "May increase the risk of high potassium levels.",
        "severity": "High", "evidence_id": "DM-LISINOPRIL-SPIRONOLACTONE-001", "source": "DailyMed",
    },
    {
        "rxcuis": {"29046", "5640"},
        "drug_a": "Lisinopril", "drug_b": "Ibuprofen",
        "concern": "May reduce blood pressure control and affect kidney function.",
        "severity": "Moderate", "evidence_id": "DM-LISINOPRIL-NSAID-001", "source": "DailyMed",
    },
    {
        "rxcuis": {"9997", "5640"},
        "drug_a": "Spironolactone", "drug_b": "Ibuprofen",
        "concern": "Concomitant NSAID use may contribute to worsening renal function.",
        "severity": "Moderate", "evidence_id": "DM-SPIRONOLACTONE-NSAID-001", "source": "DailyMed",
    },
]

_RULE_DRUG_LAB = [
    {
        "rxcui": "29046", "drug": "Lisinopril", "lab": "potassium", "operator": ">", "threshold": 5.0,
        "severity": "High", "concern": "Elevated potassium may increase the safety concern with lisinopril.",
        "threshold_basis": "Prototype threshold. DailyMed supports hyperkalemia risk and potassium monitoring but does not establish the >5.0 cutoff used by this rule.",
        "evidence_id": "DM-LISINOPRIL-HYPERKALEMIA-001", "source": "DailyMed",
    },
    {
        "rxcui": "9997", "drug": "Spironolactone", "lab": "potassium", "operator": ">", "threshold": 5.0,
        "severity": "High", "concern": "Elevated potassium may increase the safety concern with spironolactone.",
        "threshold_basis": "Prototype threshold. DailyMed supports hyperkalemia risk and potassium monitoring but does not establish the >5.0 cutoff used by this rule.",
        "evidence_id": "DM-SPIRONOLACTONE-HYPERKALEMIA-001", "source": "DailyMed",
    },
    {
        "rxcui": "5640", "drug": "Ibuprofen", "lab": "egfr", "operator": "<", "threshold": 60.0,
        "severity": "Moderate", "concern": "Reduced kidney function may increase concern with ibuprofen.",
        "threshold_basis": "Prototype threshold. DailyMed supports renal risk with ibuprofen/NSAID use but does not establish the eGFR <60 cutoff used by this rule.",
        "evidence_id": "DM-IBUPROFEN-RENAL-001", "source": "DailyMed",
    },
]


def demo_pre_add_check(candidate_rxcui: str, context: dict, labs: dict | None) -> dict:
    """Demo-mode replacement for databricks_client.check_pre_add()."""
    current_medications = context.get("current_medications") or []
    current_rxcuis = {
        ingredient.get("rxcui")
        for med in current_medications
        for ingredient in (med.get("ingredients") or [])
    }

    drug_drug_findings = []
    for rule in _RULE_DRUG_DRUG:
        if candidate_rxcui in rule["rxcuis"] and rule["rxcuis"] - {candidate_rxcui} & current_rxcuis:
            existing_rxcui = next(iter(rule["rxcuis"] - {candidate_rxcui}))
            drug_drug_findings.append({
                "finding_type": "Drug-Drug",
                "severity": rule["severity"],
                "new_medication": context.get("candidate_name"),
                "existing_medication": next(
                    (m.get("selected_name") for m in current_medications
                     if existing_rxcui in {i.get("rxcui") for i in (m.get("ingredients") or [])}),
                    None,
                ),
                "drug_a": rule["drug_a"],
                "drug_b": rule["drug_b"],
                "concern": rule["concern"],
                "evidence_id": rule["evidence_id"],
                "source": rule["source"],
                "review_status": "Prototype Reviewed",
            })

    labs = labs or {}
    drug_lab_findings = []
    for rule in _RULE_DRUG_LAB:
        if rule["rxcui"] != candidate_rxcui:
            continue
        observed = labs.get(rule["lab"])
        if observed is None:
            continue
        triggered = observed > rule["threshold"] if rule["operator"] == ">" else observed < rule["threshold"]
        if not triggered:
            continue
        drug_lab_findings.append({
            "finding_type": "Drug-Lab",
            "severity": rule["severity"],
            "drug": rule["drug"],
            "drug_rxcui": rule["rxcui"],
            "lab": rule["lab"],
            "observed_value": observed,
            "operator": rule["operator"],
            "threshold": rule["threshold"],
            "concern": rule["concern"],
            "threshold_basis": rule["threshold_basis"],
            "evidence_id": rule["evidence_id"],
            "source": rule["source"],
        })

    total_findings = len(drug_drug_findings) + len(drug_lab_findings)
    covered_drug = candidate_rxcui in {"29046", "9997", "5640"}

    if total_findings:
        message = (
            "Safety findings were identified from the current prototype rules. Review the "
            "findings before starting a proposed medication. Existing medications should still "
            "be recorded accurately."
        )
    else:
        message = (
            "No supported safety concern was identified from the current prototype rules and "
            "available patient data."
        )

    coverage_note = (
        "[DEMO MODE] Pre-add screening is using a small offline rule table (Lisinopril, "
        "Spironolactone, Ibuprofen only) replayed from a live capture — DEMO_MODE has no live "
        "openFDA fallback."
        if covered_drug
        else
        "[DEMO MODE] This drug is outside DEMO_MODE's canned rule-table coverage (Lisinopril, "
        "Spironolactone, Ibuprofen only), and DEMO_MODE has no live openFDA fallback, so no "
        "findings can be produced for it offline. Disable DEMO_MODE and configure Databricks "
        "credentials for real coverage."
    )

    return {
        "candidate_rxcui": candidate_rxcui,
        "candidate_name": context.get("candidate_name"),
        "drug_drug_findings": drug_drug_findings,
        "drug_lab_findings": drug_lab_findings,
        "total_findings": total_findings,
        "safety_review_required": total_findings > 0,
        "message": message,
        "coverage_note": coverage_note,
    }
