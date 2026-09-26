# Databricks notebook source
import requests
import json

print("Drug Data Integration notebook ready.")

# COMMAND ----------

drug_name = "Lisinopril"

url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

params = {
    "name": drug_name
}

response = requests.get(
    url,
    params=params,
    timeout=20
)

print("HTTP Status:", response.status_code)

data = response.json()

print(json.dumps(data, indent=2))

# COMMAND ----------

def get_rxcui(drug_name):
    """
    Look up a drug name in RxNorm and return its RxCUI.
    Returns None if no RxCUI is found.
    """

    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

    params = {
        "name": drug_name
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    rxnorm_ids = data.get("idGroup", {}).get("rxnormId", [])

    if rxnorm_ids:
        return rxnorm_ids[0]

    return None


print("RxNorm lookup function loaded.")

# COMMAND ----------

test_drugs = [
    "Lisinopril",
    "Spironolactone",
    "Ibuprofen"
]

for drug in test_drugs:
    rxcui = get_rxcui(drug)
    print(f"{drug}: {rxcui}")

# COMMAND ----------

def normalize_drug(drug_name):
    """
    Normalize a medication name using RxNorm.

    Returns:
        {
            "input_name": original name,
            "rxcui": RxNorm Concept Unique Identifier,
            "normalized_name": RxNorm preferred name
        }

    Returns None if the drug cannot be resolved.
    """

    # Step 1: Find RxCUI
    rxcui = get_rxcui(drug_name)

    if not rxcui:
        return None

    # Step 2: Retrieve RxNorm concept properties
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


print("Drug normalization function loaded.")

# COMMAND ----------

test_drugs = [
    "Lisinopril",
    "Spironolactone",
    "Ibuprofen"
]

for drug in test_drugs:

    normalized = normalize_drug(drug)

    print(normalized)

# COMMAND ----------

def normalize_medication_list(medications):
    """
    Normalize a list of medication names using RxNorm.
    """

    normalized_medications = []

    for drug in medications:
        normalized = normalize_drug(drug)

        if normalized:
            normalized_medications.append(normalized)
        else:
            normalized_medications.append({
                "input_name": drug,
                "rxcui": None,
                "normalized_name": None
            })

    return normalized_medications


print("Medication list normalization function loaded.")

# COMMAND ----------

patient_medications = [
    "Lisinopril",
    "Spironolactone",
    "Ibuprofen"
]

normalized_medications = normalize_medication_list(
    patient_medications
)

for medication in normalized_medications:
    print(medication)

# COMMAND ----------

test_bad_drug = normalize_drug("LisinoprilXYZ")

print(test_bad_drug)

# COMMAND ----------

patient = {
    "patient_id": "P003",
    "age": 60,
    "medications": [
        "Lisinopril",
        "Spironolactone",
        "Ibuprofen"
    ],
    "conditions": [
        "Hypertension"
    ],
    "labs": {
        "potassium": 5.2,
        "egfr": 58
    },
    "allergies": []
}

normalized_medications = normalize_medication_list(
    patient["medications"]
)

for medication in normalized_medications:
    print(medication)

# COMMAND ----------

patient["normalized_medications"] = normalized_medications

patient

# COMMAND ----------

normalized_names = [
    medication["normalized_name"]
    for medication in patient["normalized_medications"]
    if medication["normalized_name"] is not None
]

print(normalized_names)

# COMMAND ----------

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

print("Drug interaction rules loaded:", len(drug_interactions))
print("Drug-lab rules loaded:", len(lab_rules))

# COMMAND ----------

from itertools import combinations

found_interactions = []

for drug1, drug2 in combinations(normalized_names, 2):

    for interaction in drug_interactions:

        rule_drug_a = interaction["drug_a"].lower()
        rule_drug_b = interaction["drug_b"].lower()

        match = (
            drug1 == rule_drug_a
            and drug2 == rule_drug_b
        ) or (
            drug2 == rule_drug_a
            and drug1 == rule_drug_b
        )

        if match:
            found_interactions.append(interaction)

for item in found_interactions:
    print(
        item["severity"],
        "-",
        item["drug_a"],
        "+",
        item["drug_b"],
        "-",
        item["interaction"]
    )

# COMMAND ----------

drug_lab_findings = []

for rule in lab_rules:

    rule_drug = rule["drug"].lower()

    drug_present = rule_drug in normalized_names

    lab_name = rule["lab"]
    lab_value = patient["labs"].get(lab_name)

    if drug_present and lab_value is not None:

        triggered = False

        if rule["operator"] == ">" and lab_value > rule["threshold"]:
            triggered = True

        elif rule["operator"] == "<" and lab_value < rule["threshold"]:
            triggered = True

        if triggered:
            drug_lab_findings.append({
                "drug": rule["drug"],
                "lab": lab_name,
                "value": lab_value,
                "severity": rule["severity"],
                "message": rule["message"]
            })


for item in drug_lab_findings:
    print(
        item["severity"],
        "-",
        item["drug"],
        "-",
        item["lab"],
        "=",
        item["value"],
        "-",
        item["message"]
    )

# COMMAND ----------

def prepare_patient_medications(patient):
    """
    Enrich a patient record with RxNorm-normalized medications.
    """

    normalized_medications = normalize_medication_list(
        patient["medications"]
    )

    enriched_patient = patient.copy()

    enriched_patient["normalized_medications"] = normalized_medications

    return enriched_patient


print("Patient medication preparation function loaded.")

# COMMAND ----------

prepared_patient = prepare_patient_medications(patient)

print("Patient:", prepared_patient["patient_id"])

print("\nOriginal Medications:")
for drug in prepared_patient["medications"]:
    print("-", drug)

print("\nNormalized Medications:")
for drug in prepared_patient["normalized_medications"]:
    print(
        f'- {drug["input_name"]} '
        f'→ RxCUI {drug["rxcui"]} '
        f'→ {drug["normalized_name"]}'
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE healthcare.medication_safety.drug_interactions
# MAGIC ADD COLUMNS (
# MAGIC     drug_a_rxcui STRING,
# MAGIC     drug_b_rxcui STRING
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_interactions
# MAGIC SET
# MAGIC     drug_a_rxcui =
# MAGIC         CASE
# MAGIC             WHEN LOWER(drug_a) = 'lisinopril' THEN '29046'
# MAGIC             WHEN LOWER(drug_a) = 'spironolactone' THEN '9997'
# MAGIC             WHEN LOWER(drug_a) = 'ibuprofen' THEN '5640'
# MAGIC         END,
# MAGIC
# MAGIC     drug_b_rxcui =
# MAGIC         CASE
# MAGIC             WHEN LOWER(drug_b) = 'lisinopril' THEN '29046'
# MAGIC             WHEN LOWER(drug_b) = 'spironolactone' THEN '9997'
# MAGIC             WHEN LOWER(drug_b) = 'ibuprofen' THEN '5640'
# MAGIC         END;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug_a,
# MAGIC     drug_a_rxcui,
# MAGIC     drug_b,
# MAGIC     drug_b_rxcui,
# MAGIC     severity,
# MAGIC     interaction
# MAGIC FROM healthcare.medication_safety.drug_interactions;

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE healthcare.medication_safety.drug_lab_rules
# MAGIC ADD COLUMNS (
# MAGIC     drug_rxcui STRING
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_lab_rules
# MAGIC SET drug_rxcui =
# MAGIC     CASE
# MAGIC         WHEN LOWER(drug) = 'lisinopril' THEN '29046'
# MAGIC         WHEN LOWER(drug) = 'spironolactone' THEN '9997'
# MAGIC         WHEN LOWER(drug) = 'ibuprofen' THEN '5640'
# MAGIC     END;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug,
# MAGIC     drug_rxcui,
# MAGIC     lab,
# MAGIC     operator,
# MAGIC     threshold,
# MAGIC     severity,
# MAGIC     message
# MAGIC FROM healthcare.medication_safety.drug_lab_rules;

# COMMAND ----------

patient_rxcuis = [
    medication["rxcui"]
    for medication in prepared_patient["normalized_medications"]
    if medication["rxcui"] is not None
]

print("Patient RxCUIs:", patient_rxcuis)

# COMMAND ----------

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

print("Drug interaction rules:", len(drug_interactions))
print("Drug-lab rules:", len(lab_rules))

print("\nSample Drug-Drug Rule:")
print(drug_interactions[0])

print("\nSample Drug-Lab Rule:")
print(lab_rules[0])

# COMMAND ----------

from itertools import combinations

found_interactions_rxcui = []

for rxcui1, rxcui2 in combinations(patient_rxcuis, 2):

    for interaction in drug_interactions:

        rule_a = interaction["drug_a_rxcui"]
        rule_b = interaction["drug_b_rxcui"]

        match = (
            rxcui1 == rule_a and rxcui2 == rule_b
        ) or (
            rxcui2 == rule_a and rxcui1 == rule_b
        )

        if match:
            found_interactions_rxcui.append(interaction)


print("Drug-Drug findings:", len(found_interactions_rxcui))

for item in found_interactions_rxcui:
    print(
        item["severity"],
        "-",
        item["drug_a"],
        "+",
        item["drug_b"],
        "-",
        item["interaction"]
    )

# COMMAND ----------

drug_lab_findings_rxcui = []

for rule in lab_rules:

    drug_present = rule["drug_rxcui"] in patient_rxcuis

    lab_name = rule["lab"]
    lab_value = prepared_patient["labs"].get(lab_name)

    if drug_present and lab_value is not None:

        triggered = False

        if rule["operator"] == ">" and lab_value > rule["threshold"]:
            triggered = True

        elif rule["operator"] == "<" and lab_value < rule["threshold"]:
            triggered = True

        if triggered:
            drug_lab_findings_rxcui.append({
                "drug": rule["drug"],
                "drug_rxcui": rule["drug_rxcui"],
                "lab": lab_name,
                "value": lab_value,
                "severity": rule["severity"],
                "message": rule["message"]
            })


print("Drug-Lab findings:", len(drug_lab_findings_rxcui))

for item in drug_lab_findings_rxcui:
    print(
        item["severity"],
        "-",
        item["drug"],
        f'(RxCUI {item["drug_rxcui"]})',
        "-",
        item["lab"],
        "=",
        item["value"],
        "-",
        item["message"]
    )

# COMMAND ----------

import requests
import json

def search_dailymed(drug_name, page=1, page_size=10):
    """
    Search DailyMed for drug labels by drug name.
    """

    url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

    params = {
        "drug_name": drug_name,
        "page": page,
        "pagesize": page_size
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


print("DailyMed search function loaded.")

# COMMAND ----------

dailymed_results = search_dailymed("Lisinopril")

print(json.dumps(dailymed_results, indent=2))

# COMMAND ----------

def summarize_dailymed_results(dailymed_results):
    """
    Extract the most useful metadata from DailyMed search results.
    """

    summaries = []

    data = dailymed_results.get("data", [])

    for item in data:
        summaries.append({
            "setid": item.get("setid"),
            "title": item.get("title"),
            "published_date": item.get("published_date")
        })

    return summaries


label_summaries = summarize_dailymed_results(dailymed_results)

print("DailyMed labels found:", len(label_summaries))

for label in label_summaries[:10]:
    print("\nSET ID:", label["setid"])
    print("TITLE:", label["title"])
    print("PUBLISHED:", label["published_date"])

# COMMAND ----------

selected_label = label_summaries[0]

selected_setid = selected_label["setid"]

print("Selected DailyMed Label")
print("-----------------------")
print("SET ID:", selected_setid)
print("TITLE:", selected_label["title"])
print("PUBLISHED:", selected_label["published_date"])

# COMMAND ----------

def get_dailymed_spl_xml(setid):
    """
    Retrieve the full DailyMed SPL document as XML.
    """

    url = (
        f"https://dailymed.nlm.nih.gov/dailymed/services/v2/"
        f"spls/{setid}.xml"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.text


spl_xml = get_dailymed_spl_xml(selected_setid)

print("Characters retrieved:", len(spl_xml))
print(spl_xml[:3000])

# COMMAND ----------

import xml.etree.ElementTree as ET

root = ET.fromstring(spl_xml)

print("SPL XML parsed successfully.")
print("Root tag:", root.tag)

# COMMAND ----------

def extract_dailymed_sections(spl_xml):
    """
    Extract labeled sections from a DailyMed SPL XML document.
    """

    root = ET.fromstring(spl_xml)

    ns = {
        "hl7": "urn:hl7-org:v3"
    }

    extracted_sections = []

    for section in root.findall(".//hl7:section", ns):

        # Section title
        title_element = section.find("hl7:title", ns)

        title = (
            "".join(title_element.itertext()).strip()
            if title_element is not None
            else None
        )

        # LOINC / section code
        code_element = section.find("hl7:code", ns)

        code = (
            code_element.get("code")
            if code_element is not None
            else None
        )

        # Section text
        text_element = section.find("hl7:text", ns)

        text = None

        if text_element is not None:
            text = " ".join(
                " ".join(text_element.itertext()).split()
            )

        extracted_sections.append({
            "title": title,
            "code": code,
            "text": text
        })

    return extracted_sections


dailymed_sections = extract_dailymed_sections(spl_xml)

print("Sections extracted:", len(dailymed_sections))

# COMMAND ----------

for index, section in enumerate(dailymed_sections, start=1):

    print(
        index,
        "|",
        section["title"],
        "| Code:",
        section["code"],
        "| Characters:",
        len(section["text"]) if section["text"] else 0
    )

# COMMAND ----------

SAFETY_SECTION_KEYWORDS = [
    "CONTRAINDICATION",
    "WARNING",
    "PRECAUTION",
    "DRUG INTERACTION",
    "ADVERSE REACTION",
    "USE IN SPECIFIC POPULATIONS"
]


safety_sections = []

for section in dailymed_sections:

    title = section["title"] or ""

    if any(
        keyword in title.upper()
        for keyword in SAFETY_SECTION_KEYWORDS
    ):
        safety_sections.append(section)


print("Safety sections found:", len(safety_sections))

for section in safety_sections:

    print("\n" + "=" * 70)
    print(section["title"])
    print("Code:", section["code"])
    print("=" * 70)

    # Only show first 1,000 characters for now
    print((section["text"] or "")[:1000])

# COMMAND ----------

for index, label in enumerate(label_summaries, start=1):

    print(
        index,
        "|",
        label["title"],
        "|",
        label["setid"],
        "|",
        label["published_date"]
    )

# COMMAND ----------

def select_single_ingredient_label(label_summaries, drug_name):
    """
    Select a likely single-ingredient DailyMed label.

    Prototype selection:
    1. Must contain requested drug name.
    2. Exclude obvious combination products.
    3. Prefer the most recently published eligible label.
    """

    candidates = []

    drug_name_lower = drug_name.lower()

    for label in label_summaries:

        title = (label["title"] or "").lower()

        if drug_name_lower not in title:
            continue

        # Exclude obvious combination products
        if " and " in title:
            continue

        candidates.append(label)

    if not candidates:
        return None

    # DailyMed results are currently returned newest-first,
    # so use the first eligible candidate.
    return candidates[0]

# COMMAND ----------

selected_label = select_single_ingredient_label(
    label_summaries,
    "Lisinopril"
)

print("Selected label:")
print(selected_label["title"])
print("SET ID:", selected_label["setid"])
print("Published:", selected_label["published_date"])

# COMMAND ----------

selected_setid = selected_label["setid"]

spl_xml = get_dailymed_spl_xml(selected_setid)

dailymed_sections = extract_dailymed_sections(spl_xml)

print("Label:", selected_label["title"])
print("SET ID:", selected_setid)
print("Sections extracted:", len(dailymed_sections))

# COMMAND ----------

for index, section in enumerate(dailymed_sections, start=1):
    print(
        index,
        "|",
        section["title"],
        "| Code:",
        section["code"],
        "| Characters:",
        len(section["text"]) if section["text"] else 0
    )

# COMMAND ----------

def get_section_group(sections, parent_title_prefix):
    """
    Return a parent section plus all following numbered child sections
    until the next major numbered section begins.

    Example:
        parent_title_prefix = "7 DRUG INTERACTIONS"
    """

    grouped_sections = []
    collecting = False
    parent_number = None

    for section in sections:

        title = section["title"] or ""

        # Find the parent section
        if title.upper().startswith(parent_title_prefix.upper()):

            collecting = True

            first_token = title.split()[0]

            if first_token.isdigit():
                parent_number = first_token

            grouped_sections.append(section)
            continue


        if collecting:

            first_token = title.split()[0] if title else ""

            # Child section such as 7.1, 7.2, 7.3
            if (
                parent_number
                and first_token.startswith(parent_number + ".")
            ):
                grouped_sections.append(section)
                continue


            # Stop when the next major numbered section begins
            if first_token.isdigit():
                break

    return grouped_sections


print("Hierarchical section grouping function loaded.")

# COMMAND ----------

drug_interaction_sections = get_section_group(
    dailymed_sections,
    "7 DRUG INTERACTIONS"
)

print(
    "Drug interaction sections found:",
    len(drug_interaction_sections)
)

for section in drug_interaction_sections:

    print(
        "-",
        section["title"],
        "| Characters:",
        len(section["text"]) if section["text"] else 0
    )

# COMMAND ----------

def combine_section_text(section_group):
    """
    Combine parent/child DailyMed sections into one readable text block.
    """

    combined_parts = []

    for section in section_group:

        title = section["title"]
        text = section["text"]

        if title:
            combined_parts.append(f"\n{title}")

        if text:
            combined_parts.append(text)

    return "\n".join(combined_parts).strip()


drug_interaction_text = combine_section_text(
    drug_interaction_sections
)

print(drug_interaction_text[:5000])

# COMMAND ----------

def find_dailymed_subsection(sections, title_contains):
    """
    Find DailyMed subsections whose title contains the requested text.
    """

    matches = []

    search_text = title_contains.lower()

    for section in sections:

        title = section["title"] or ""

        if search_text in title.lower():
            matches.append(section)

    return matches


nsaid_sections = find_dailymed_subsection(
    dailymed_sections,
    "Non-Steroidal Anti-Inflammatory"
)

print("Matches found:", len(nsaid_sections))

for section in nsaid_sections:
    print("\nTITLE:")
    print(section["title"])

    print("\nTEXT:")
    print(section["text"])

# COMMAND ----------

lisinopril_nsaid_evidence = {
    "drug": "Lisinopril",
    "drug_rxcui": "29046",

    "interaction_class": "NSAID",

    "source": "DailyMed",
    "setid": selected_setid,

    "section_code": "34073-7",
    "section_title": "DRUG INTERACTIONS",

    "subsection_title": nsaid_sections[0]["title"],
    "evidence_text": nsaid_sections[0]["text"]
}

print(json.dumps(
    lisinopril_nsaid_evidence,
    indent=2
))

# COMMAND ----------

print("Drug:", lisinopril_nsaid_evidence["drug"])
print("RxCUI:", lisinopril_nsaid_evidence["drug_rxcui"])
print("Source:", lisinopril_nsaid_evidence["source"])
print("SPL SET ID:", lisinopril_nsaid_evidence["setid"])
print("Subsection:", lisinopril_nsaid_evidence["subsection_title"])

print("\nEvidence:")
print(lisinopril_nsaid_evidence["evidence_text"])

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS healthcare.medication_safety.source_evidence (
# MAGIC     evidence_id STRING,
# MAGIC     drug STRING,
# MAGIC     drug_rxcui STRING,
# MAGIC     interaction_class STRING,
# MAGIC     source STRING,
# MAGIC     source_setid STRING,
# MAGIC     section_code STRING,
# MAGIC     section_title STRING,
# MAGIC     subsection_title STRING,
# MAGIC     evidence_text STRING,
# MAGIC     reviewed BOOLEAN,
# MAGIC     created_at TIMESTAMP
# MAGIC );

# COMMAND ----------

from datetime import datetime

evidence_row = [{
    "evidence_id": "DM-LISINOPRIL-NSAID-001",
    "drug": "Lisinopril",
    "drug_rxcui": "29046",
    "interaction_class": "NSAID",
    "source": "DailyMed",
    "source_setid": selected_setid,
    "section_code": "34073-7",
    "section_title": "DRUG INTERACTIONS",
    "subsection_title": nsaid_sections[0]["title"],
    "evidence_text": nsaid_sections[0]["text"],
    "reviewed": False,
    "created_at": datetime.now()
}]

evidence_df = spark.createDataFrame(evidence_row)

evidence_df.write.mode("append").saveAsTable(
    "healthcare.medication_safety.source_evidence"
)

print("DailyMed evidence stored.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     evidence_id,
# MAGIC     drug,
# MAGIC     drug_rxcui,
# MAGIC     interaction_class,
# MAGIC     source,
# MAGIC     source_setid,
# MAGIC     subsection_title,
# MAGIC     reviewed
# MAGIC FROM healthcare.medication_safety.source_evidence;

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE healthcare.medication_safety.drug_interactions
# MAGIC ADD COLUMNS (
# MAGIC     evidence_id STRING,
# MAGIC     source STRING,
# MAGIC     review_status STRING
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_interactions
# MAGIC SET
# MAGIC     evidence_id = 'DM-LISINOPRIL-NSAID-001',
# MAGIC     source = 'DailyMed',
# MAGIC     review_status = 'Prototype Reviewed'
# MAGIC WHERE
# MAGIC     (
# MAGIC         drug_a_rxcui = '29046'
# MAGIC         AND drug_b_rxcui = '5640'
# MAGIC     )
# MAGIC     OR
# MAGIC     (
# MAGIC         drug_a_rxcui = '5640'
# MAGIC         AND drug_b_rxcui = '29046'
# MAGIC     );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug_a,
# MAGIC     drug_a_rxcui,
# MAGIC     drug_b,
# MAGIC     drug_b_rxcui,
# MAGIC     severity,
# MAGIC     interaction,
# MAGIC     evidence_id,
# MAGIC     source,
# MAGIC     review_status
# MAGIC FROM healthcare.medication_safety.drug_interactions;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     r.drug_a,
# MAGIC     r.drug_b,
# MAGIC     r.severity,
# MAGIC     r.interaction,
# MAGIC
# MAGIC     e.source,
# MAGIC     e.source_setid,
# MAGIC     e.subsection_title,
# MAGIC     e.evidence_text
# MAGIC
# MAGIC FROM healthcare.medication_safety.drug_interactions r
# MAGIC
# MAGIC LEFT JOIN healthcare.medication_safety.source_evidence e
# MAGIC     ON r.evidence_id = e.evidence_id
# MAGIC
# MAGIC WHERE r.evidence_id = 'DM-LISINOPRIL-NSAID-001';

# COMMAND ----------

def search_dailymed_sections(sections, search_terms):
    """
    Search DailyMed section titles and text for one or more terms.
    """

    matches = []

    search_terms = [
        term.lower()
        for term in search_terms
    ]

    for section in sections:

        title = section["title"] or ""
        text = section["text"] or ""

        searchable_text = (
            title + " " + text
        ).lower()

        matched_terms = [
            term
            for term in search_terms
            if term in searchable_text
        ]

        if matched_terms:
            matches.append({
                "title": section["title"],
                "code": section["code"],
                "text": section["text"],
                "matched_terms": matched_terms
            })

    return matches


spironolactone_matches = search_dailymed_sections(
    dailymed_sections,
    [
        "spironolactone",
        "potassium-sparing"
    ]
)

print("Matches found:", len(spironolactone_matches))

for match in spironolactone_matches:

    print("\n" + "=" * 70)
    print("TITLE:", match["title"])
    print("CODE:", match["code"])
    print("MATCHED:", match["matched_terms"])
    print("=" * 70)

    print((match["text"] or "")[:2500])

# COMMAND ----------

spironolactone_evidence_section = next(
    match
    for match in spironolactone_matches
    if match["title"] == "7.1 Diuretics"
)

lisinopril_spironolactone_evidence = {
    "evidence_id": "DM-LISINOPRIL-SPIRONOLACTONE-001",

    "drug": "Lisinopril",
    "drug_rxcui": "29046",

    "interaction_class": "Potassium-sparing diuretic",

    "source": "DailyMed",
    "source_setid": selected_setid,

    "section_code": "34073-7",
    "section_title": "DRUG INTERACTIONS",

    "subsection_title": spironolactone_evidence_section["title"],
    "evidence_text": spironolactone_evidence_section["text"],

    "reviewed": False
}

print(json.dumps(
    lisinopril_spironolactone_evidence,
    indent=2
))

# COMMAND ----------

from datetime import datetime

evidence_row = [{
    "evidence_id":
        lisinopril_spironolactone_evidence["evidence_id"],

    "drug":
        lisinopril_spironolactone_evidence["drug"],

    "drug_rxcui":
        lisinopril_spironolactone_evidence["drug_rxcui"],

    "interaction_class":
        lisinopril_spironolactone_evidence["interaction_class"],

    "source":
        lisinopril_spironolactone_evidence["source"],

    "source_setid":
        lisinopril_spironolactone_evidence["source_setid"],

    "section_code":
        lisinopril_spironolactone_evidence["section_code"],

    "section_title":
        lisinopril_spironolactone_evidence["section_title"],

    "subsection_title":
        lisinopril_spironolactone_evidence["subsection_title"],

    "evidence_text":
        lisinopril_spironolactone_evidence["evidence_text"],

    "reviewed": False,

    "created_at": datetime.now()
}]

spark.createDataFrame(
    evidence_row
).write.mode("append").saveAsTable(
    "healthcare.medication_safety.source_evidence"
)

print("Lisinopril + Spironolactone evidence stored.")

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_interactions
# MAGIC SET
# MAGIC     evidence_id = 'DM-LISINOPRIL-SPIRONOLACTONE-001',
# MAGIC     source = 'DailyMed',
# MAGIC     review_status = 'Prototype Reviewed'
# MAGIC WHERE
# MAGIC     (
# MAGIC         drug_a_rxcui = '29046'
# MAGIC         AND drug_b_rxcui = '9997'
# MAGIC     )
# MAGIC     OR
# MAGIC     (
# MAGIC         drug_a_rxcui = '9997'
# MAGIC         AND drug_b_rxcui = '29046'
# MAGIC     );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug_a,
# MAGIC     drug_b,
# MAGIC     severity,
# MAGIC     interaction,
# MAGIC     evidence_id,
# MAGIC     source,
# MAGIC     review_status
# MAGIC FROM healthcare.medication_safety.drug_interactions;

# COMMAND ----------

spironolactone_results = search_dailymed(
    "Spironolactone"
)

spironolactone_summaries = summarize_dailymed_results(
    spironolactone_results
)

print(
    "Labels returned:",
    len(spironolactone_summaries)
)

for i, label in enumerate(
    spironolactone_summaries,
    start=1
):

    print(
        i,
        "|",
        label["title"],
        "|",
        label["setid"],
        "|",
        label["published_date"]
    )

# COMMAND ----------

spironolactone_selected_setid = (
    spironolactone_summaries[0]["setid"]
)

print(
    "Selected label:",
    spironolactone_summaries[0]["title"]
)

print(
    "SET ID:",
    spironolactone_selected_setid
)


spironolactone_spl_xml = get_dailymed_spl_xml(
    spironolactone_selected_setid
)

spironolactone_sections = extract_dailymed_sections(
    spironolactone_spl_xml
)

print(
    "Sections extracted:",
    len(spironolactone_sections)
)

# COMMAND ----------

spironolactone_nsaid_matches = search_dailymed_sections(
    spironolactone_sections,
    [
        "ibuprofen",
        "nsaid",
        "non-steroidal",
        "nonsteroidal",
        "renal function"
    ]
)

print(
    "Matches found:",
    len(spironolactone_nsaid_matches)
)

for match in spironolactone_nsaid_matches:

    print("\n" + "=" * 70)

    print(
        "TITLE:",
        match["title"]
    )

    print(
        "CODE:",
        match["code"]
    )

    print(
        "MATCHED:",
        match["matched_terms"]
    )

    print("=" * 70)

    print(
        (match["text"] or "")[:3000]
    )

# COMMAND ----------

spironolactone_nsaid_evidence_section = next(
    match
    for match in spironolactone_nsaid_matches
    if match["title"] == "5.2 Hypotension and Worsening Renal Function"
)

spironolactone_ibuprofen_evidence = {
    "evidence_id": "DM-SPIRONOLACTONE-NSAID-001",

    "drug": "Spironolactone",
    "drug_rxcui": "9997",

    "interaction_class": "NSAID",

    "source": "DailyMed",
    "source_setid": spironolactone_selected_setid,

    "section_code": "42229-5",
    "section_title": "WARNINGS AND PRECAUTIONS",

    "subsection_title":
        spironolactone_nsaid_evidence_section["title"],

    "evidence_text":
        spironolactone_nsaid_evidence_section["text"],

    "reviewed": False
}

print(
    json.dumps(
        spironolactone_ibuprofen_evidence,
        indent=2
    )
)

# COMMAND ----------

from datetime import datetime

evidence_row = [{
    "evidence_id":
        spironolactone_ibuprofen_evidence["evidence_id"],

    "drug":
        spironolactone_ibuprofen_evidence["drug"],

    "drug_rxcui":
        spironolactone_ibuprofen_evidence["drug_rxcui"],

    "interaction_class":
        spironolactone_ibuprofen_evidence["interaction_class"],

    "source":
        spironolactone_ibuprofen_evidence["source"],

    "source_setid":
        spironolactone_ibuprofen_evidence["source_setid"],

    "section_code":
        spironolactone_ibuprofen_evidence["section_code"],

    "section_title":
        spironolactone_ibuprofen_evidence["section_title"],

    "subsection_title":
        spironolactone_ibuprofen_evidence["subsection_title"],

    "evidence_text":
        spironolactone_ibuprofen_evidence["evidence_text"],

    "reviewed": False,

    "created_at": datetime.now()
}]

spark.createDataFrame(
    evidence_row
).write.mode("append").saveAsTable(
    "healthcare.medication_safety.source_evidence"
)

print(
    "Spironolactone + NSAID evidence stored."
)

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_interactions
# MAGIC SET
# MAGIC     interaction = 'Concomitant NSAID use may contribute to worsening renal function.',
# MAGIC     evidence_id = 'DM-SPIRONOLACTONE-NSAID-001',
# MAGIC     source = 'DailyMed',
# MAGIC     review_status = 'Prototype Reviewed'
# MAGIC WHERE
# MAGIC     (
# MAGIC         drug_a_rxcui = '9997'
# MAGIC         AND drug_b_rxcui = '5640'
# MAGIC     )
# MAGIC     OR
# MAGIC     (
# MAGIC         drug_a_rxcui = '5640'
# MAGIC         AND drug_b_rxcui = '9997'
# MAGIC     );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug_a,
# MAGIC     drug_b,
# MAGIC     severity,
# MAGIC     interaction,
# MAGIC     evidence_id,
# MAGIC     source,
# MAGIC     review_status
# MAGIC FROM healthcare.medication_safety.drug_interactions;

# COMMAND ----------

spironolactone_hyperkalemia_section = next(
    match
    for match in spironolactone_nsaid_matches
    if match["title"] == "5.1 Hyperkalemia"
)

spironolactone_potassium_evidence = {
    "evidence_id": "DM-SPIRONOLACTONE-HYPERKALEMIA-001",

    "drug": "Spironolactone",
    "drug_rxcui": "9997",

    "interaction_class": "Drug-Lab: Potassium",

    "source": "DailyMed",
    "source_setid": spironolactone_selected_setid,

    "section_code": "42229-5",
    "section_title": "WARNINGS AND PRECAUTIONS",

    "subsection_title":
        spironolactone_hyperkalemia_section["title"],

    "evidence_text":
        spironolactone_hyperkalemia_section["text"],

    "reviewed": False
}

print(
    json.dumps(
        spironolactone_potassium_evidence,
        indent=2
    )
)

# COMMAND ----------

from datetime import datetime

evidence_row = [{
    "evidence_id":
        spironolactone_potassium_evidence["evidence_id"],

    "drug":
        spironolactone_potassium_evidence["drug"],

    "drug_rxcui":
        spironolactone_potassium_evidence["drug_rxcui"],

    "interaction_class":
        spironolactone_potassium_evidence["interaction_class"],

    "source":
        spironolactone_potassium_evidence["source"],

    "source_setid":
        spironolactone_potassium_evidence["source_setid"],

    "section_code":
        spironolactone_potassium_evidence["section_code"],

    "section_title":
        spironolactone_potassium_evidence["section_title"],

    "subsection_title":
        spironolactone_potassium_evidence["subsection_title"],

    "evidence_text":
        spironolactone_potassium_evidence["evidence_text"],

    "reviewed": False,

    "created_at": datetime.now()
}]

spark.createDataFrame(
    evidence_row
).write.mode("append").saveAsTable(
    "healthcare.medication_safety.source_evidence"
)

print("Spironolactone potassium evidence stored.")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE healthcare.medication_safety.drug_lab_rules
# MAGIC ADD COLUMNS (
# MAGIC     evidence_id STRING,
# MAGIC     source STRING,
# MAGIC     review_status STRING,
# MAGIC     threshold_basis STRING
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_lab_rules
# MAGIC SET
# MAGIC     evidence_id = 'DM-SPIRONOLACTONE-HYPERKALEMIA-001',
# MAGIC     source = 'DailyMed',
# MAGIC     review_status = 'Prototype Reviewed',
# MAGIC     threshold_basis =
# MAGIC         'Prototype threshold. DailyMed supports hyperkalemia risk and potassium monitoring but does not establish the >5.0 cutoff used by this rule.'
# MAGIC WHERE
# MAGIC     drug_rxcui = '9997'
# MAGIC     AND LOWER(lab) = 'potassium';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug,
# MAGIC     lab,
# MAGIC     operator,
# MAGIC     threshold,
# MAGIC     severity,
# MAGIC     evidence_id,
# MAGIC     source,
# MAGIC     review_status,
# MAGIC     threshold_basis
# MAGIC FROM healthcare.medication_safety.drug_lab_rules;

# COMMAND ----------

lisinopril_hyperkalemia_sections = find_dailymed_subsection(
    dailymed_sections,
    "5.5 Hyperkalemia"
)

print(
    "Matches found:",
    len(lisinopril_hyperkalemia_sections)
)

for section in lisinopril_hyperkalemia_sections:

    print("\n" + "=" * 70)
    print("TITLE:", section["title"])
    print("CODE:", section["code"])
    print("=" * 70)
    print(section["text"])

# COMMAND ----------

lisinopril_hyperkalemia_section = (
    lisinopril_hyperkalemia_sections[0]
)

lisinopril_potassium_evidence = {
    "evidence_id": "DM-LISINOPRIL-HYPERKALEMIA-001",

    "drug": "Lisinopril",
    "drug_rxcui": "29046",

    "interaction_class": "Drug-Lab: Potassium",

    "source": "DailyMed",
    "source_setid": selected_setid,

    "section_code": "43685-7",
    "section_title": "WARNINGS AND PRECAUTIONS",

    "subsection_title":
        lisinopril_hyperkalemia_section["title"],

    "evidence_text":
        lisinopril_hyperkalemia_section["text"],

    "reviewed": False
}

print(
    json.dumps(
        lisinopril_potassium_evidence,
        indent=2
    )
)

# COMMAND ----------

from datetime import datetime

evidence_row = [{
    "evidence_id":
        lisinopril_potassium_evidence["evidence_id"],

    "drug":
        lisinopril_potassium_evidence["drug"],

    "drug_rxcui":
        lisinopril_potassium_evidence["drug_rxcui"],

    "interaction_class":
        lisinopril_potassium_evidence["interaction_class"],

    "source":
        lisinopril_potassium_evidence["source"],

    "source_setid":
        lisinopril_potassium_evidence["source_setid"],

    "section_code":
        lisinopril_potassium_evidence["section_code"],

    "section_title":
        lisinopril_potassium_evidence["section_title"],

    "subsection_title":
        lisinopril_potassium_evidence["subsection_title"],

    "evidence_text":
        lisinopril_potassium_evidence["evidence_text"],

    "reviewed": False,

    "created_at": datetime.now()
}]

spark.createDataFrame(
    evidence_row
).write.mode("append").saveAsTable(
    "healthcare.medication_safety.source_evidence"
)

print("Lisinopril potassium evidence stored.")

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_lab_rules
# MAGIC SET
# MAGIC     evidence_id = 'DM-LISINOPRIL-HYPERKALEMIA-001',
# MAGIC     source = 'DailyMed',
# MAGIC     review_status = 'Prototype Reviewed',
# MAGIC     threshold_basis =
# MAGIC         'Prototype threshold. DailyMed supports hyperkalemia risk and potassium monitoring but does not establish the >5.0 cutoff used by this rule.'
# MAGIC WHERE
# MAGIC     drug_rxcui = '29046'
# MAGIC     AND LOWER(lab) = 'potassium';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug,
# MAGIC     lab,
# MAGIC     operator,
# MAGIC     threshold,
# MAGIC     severity,
# MAGIC     evidence_id,
# MAGIC     source,
# MAGIC     review_status,
# MAGIC     threshold_basis
# MAGIC FROM healthcare.medication_safety.drug_lab_rules;

# COMMAND ----------

ibuprofen_results = search_dailymed(
    "Ibuprofen"
)

ibuprofen_summaries = summarize_dailymed_results(
    ibuprofen_results
)

print(
    "Labels returned:",
    len(ibuprofen_summaries)
)

for i, label in enumerate(
    ibuprofen_summaries,
    start=1
):

    print(
        i,
        "|",
        label["title"],
        "|",
        label["setid"],
        "|",
        label["published_date"]
    )

# COMMAND ----------

ibuprofen_selected_setid = (
    ibuprofen_summaries[0]["setid"]
)

print(
    "Selected label:",
    ibuprofen_summaries[0]["title"]
)

print(
    "SET ID:",
    ibuprofen_selected_setid
)

ibuprofen_spl_xml = get_dailymed_spl_xml(
    ibuprofen_selected_setid
)

ibuprofen_sections = extract_dailymed_sections(
    ibuprofen_spl_xml
)

print(
    "Sections extracted:",
    len(ibuprofen_sections)
)

# COMMAND ----------

ibuprofen_renal_matches = search_dailymed_sections(
    ibuprofen_sections,
    [
        "renal",
        "renal impairment",
        "renal function",
        "kidney",
        "advanced renal disease",
        "creatinine",
        "glomerular"
    ]
)

print(
    "Matches found:",
    len(ibuprofen_renal_matches)
)

for match in ibuprofen_renal_matches:

    print("\n" + "=" * 70)
    print("TITLE:", match["title"])
    print("CODE:", match["code"])
    print("MATCHED:", match["matched_terms"])
    print("=" * 70)

    print(
        (match["text"] or "")[:3000]
    )

# COMMAND ----------

ibuprofen_renal_evidence_section = (
    ibuprofen_renal_matches[0]
)

ibuprofen_renal_evidence = {
    "evidence_id": "DM-IBUPROFEN-RENAL-001",

    "drug": "Ibuprofen",
    "drug_rxcui": "5640",

    "interaction_class": "Drug-Lab: Renal Function",

    "source": "DailyMed",
    "source_setid": ibuprofen_selected_setid,

    "section_code":
        ibuprofen_renal_evidence_section["code"],

    "section_title":
        "WARNINGS",

    "subsection_title":
        "Renal Effects / Advanced Renal Disease",

    "evidence_text":
        ibuprofen_renal_evidence_section["text"],

    "reviewed": False
}

print(
    json.dumps(
        ibuprofen_renal_evidence,
        indent=2
    )
)

# COMMAND ----------

from datetime import datetime

evidence_row = [{
    "evidence_id":
        ibuprofen_renal_evidence["evidence_id"],

    "drug":
        ibuprofen_renal_evidence["drug"],

    "drug_rxcui":
        ibuprofen_renal_evidence["drug_rxcui"],

    "interaction_class":
        ibuprofen_renal_evidence["interaction_class"],

    "source":
        ibuprofen_renal_evidence["source"],

    "source_setid":
        ibuprofen_renal_evidence["source_setid"],

    "section_code":
        ibuprofen_renal_evidence["section_code"],

    "section_title":
        ibuprofen_renal_evidence["section_title"],

    "subsection_title":
        ibuprofen_renal_evidence["subsection_title"],

    "evidence_text":
        ibuprofen_renal_evidence["evidence_text"],

    "reviewed": False,

    "created_at": datetime.now()
}]

spark.createDataFrame(
    evidence_row
).write.mode("append").saveAsTable(
    "healthcare.medication_safety.source_evidence"
)

print("Ibuprofen renal evidence stored.")

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE healthcare.medication_safety.drug_lab_rules
# MAGIC SET
# MAGIC     evidence_id = 'DM-IBUPROFEN-RENAL-001',
# MAGIC     source = 'DailyMed',
# MAGIC     review_status = 'Prototype Reviewed',
# MAGIC     threshold_basis =
# MAGIC         'Prototype threshold. DailyMed supports renal risk with ibuprofen/NSAID use but does not establish the eGFR <60 cutoff used by this rule.'
# MAGIC WHERE
# MAGIC     drug_rxcui = '5640'
# MAGIC     AND LOWER(lab) = 'egfr';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     drug,
# MAGIC     lab,
# MAGIC     operator,
# MAGIC     threshold,
# MAGIC     severity,
# MAGIC     evidence_id,
# MAGIC     source,
# MAGIC     review_status,
# MAGIC     threshold_basis
# MAGIC FROM healthcare.medication_safety.drug_lab_rules;