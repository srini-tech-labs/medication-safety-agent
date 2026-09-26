"""Medication Management: deterministic CRUD/lookup, running locally in FastAPI.

Ported from the retired medication-lookup-api Model Serving endpoint — see
docs/databricks-endpoint-lifecycle.md for why. The one genuinely clinical
operation (pre_add_check) still runs centrally in Databricks via
medication-safety-agent (app/services/databricks_client.py:check_pre_add);
everything else here is deterministic SQL/RxNorm access.
"""

import json
import time
import uuid

from app.config import settings
from app.services import rxnorm_client
from app.services.databricks_client import DatabricksClientError, check_pre_add
from app.services.rxnorm_client import RxNormError
from app.services.sql_client import SqlClientError, execute_sql

_TABLE_PATIENT_MEDICATIONS = f"{settings.unity_catalog}.{settings.unity_schema}.patient_medications"
_TABLE_MEDICATION_CACHE = f"{settings.unity_catalog}.{settings.unity_schema}.rxnorm_medication_cache"
_TABLE_DRUG_INTERACTIONS = f"{settings.unity_catalog}.{settings.unity_schema}.drug_interactions"
_TABLE_SOURCE_EVIDENCE = f"{settings.unity_catalog}.{settings.unity_schema}.source_evidence"

_DETAIL_CACHE_TTL_SECONDS = 60 * 60 * 24
_DURABLE_DETAIL_CACHE_DAYS = 1
_detail_memory_cache: dict[str, dict] = {}


class MedicationManagementError(Exception):
    """Raised when a Medication Management operation fails."""


def _wrap(exc: Exception, action: str) -> MedicationManagementError:
    return MedicationManagementError(f"{action} failed: {exc}")


# ============================================================
# HEALTH
# ============================================================


def health() -> dict:
    try:
        rxnorm_version = rxnorm_client.rxnorm_version()
    except RxNormError as exc:
        raise _wrap(exc, "health") from exc
    try:
        sql_status = execute_sql("SELECT 'OK' AS sql_status", expect_rows=True)
    except SqlClientError as exc:
        raise _wrap(exc, "health") from exc
    return {
        "status": "OK",
        "rxnorm": rxnorm_version,
        "databricks_sql": sql_status,
        "service": "medication-management (local)",
    }


# ============================================================
# SEARCH
# ============================================================


def search_medications(query: str, limit: int | None = None) -> dict:
    try:
        return rxnorm_client.search_medications(query, limit)
    except RxNormError as exc:
        raise _wrap(exc, "search_medications") from exc


# ============================================================
# MEDICATION DETAILS (memory cache -> durable Delta cache -> live RxNorm)
# ============================================================


def _load_durable_cache(rxcui: str) -> dict | None:
    rows = execute_sql(
        f"""
        SELECT selected_rxcui, selected_name, full_name, full_generic_name, strength, route,
               rxterms_dose_form, rxnorm_dose_form, term_type, generic_rxcui,
               primary_ingredient_rxcui, primary_ingredient_name, ingredients_json, source
        FROM {_TABLE_MEDICATION_CACHE}
        WHERE selected_rxcui = :rxcui AND expires_at > current_timestamp()
        ORDER BY retrieved_at DESC
        LIMIT 1
        """,
        parameters={"rxcui": rxcui},
        expect_rows=True,
    )
    if not rows:
        return None
    row = rows[0]
    try:
        ingredients = json.loads(row.get("ingredients_json") or "[]")
    except (TypeError, ValueError):
        ingredients = []
    return {
        "selected_rxcui": row.get("selected_rxcui"),
        "selected_name": row.get("selected_name"),
        "full_name": row.get("full_name"),
        "full_generic_name": row.get("full_generic_name"),
        "strength": row.get("strength"),
        "route": row.get("route"),
        "rxterms_dose_form": row.get("rxterms_dose_form"),
        "rxnorm_dose_form": row.get("rxnorm_dose_form"),
        "term_type": row.get("term_type"),
        "generic_rxcui": row.get("generic_rxcui"),
        "primary_ingredient_rxcui": row.get("primary_ingredient_rxcui"),
        "primary_ingredient_name": row.get("primary_ingredient_name"),
        "ingredients": ingredients,
        "medication_type": "UNKNOWN",
        "source": row.get("source"),
        "cache": "delta",
    }


def _save_durable_cache(details: dict) -> None:
    execute_sql(
        f"""
        MERGE INTO {_TABLE_MEDICATION_CACHE} AS target
        USING (SELECT :selected_rxcui AS selected_rxcui) AS source
        ON target.selected_rxcui = source.selected_rxcui
        WHEN MATCHED THEN UPDATE SET
            selected_name = :selected_name, full_name = :full_name,
            full_generic_name = :full_generic_name, strength = :strength, route = :route,
            rxterms_dose_form = :rxterms_dose_form, rxnorm_dose_form = :rxnorm_dose_form,
            term_type = :term_type, generic_rxcui = :generic_rxcui,
            primary_ingredient_rxcui = :primary_ingredient_rxcui,
            primary_ingredient_name = :primary_ingredient_name, ingredients_json = :ingredients_json,
            source = :source, retrieved_at = current_timestamp(),
            expires_at = current_timestamp() + INTERVAL {_DURABLE_DETAIL_CACHE_DAYS} DAYS
        WHEN NOT MATCHED THEN INSERT (
            selected_rxcui, selected_name, full_name, full_generic_name, strength, route,
            rxterms_dose_form, rxnorm_dose_form, term_type, generic_rxcui,
            primary_ingredient_rxcui, primary_ingredient_name, ingredients_json, source,
            retrieved_at, expires_at
        ) VALUES (
            :selected_rxcui, :selected_name, :full_name, :full_generic_name, :strength, :route,
            :rxterms_dose_form, :rxnorm_dose_form, :term_type, :generic_rxcui,
            :primary_ingredient_rxcui, :primary_ingredient_name, :ingredients_json, :source,
            current_timestamp(), current_timestamp() + INTERVAL {_DURABLE_DETAIL_CACHE_DAYS} DAYS
        )
        """,
        parameters={
            "selected_rxcui": details.get("selected_rxcui"),
            "selected_name": details.get("selected_name"),
            "full_name": details.get("full_name"),
            "full_generic_name": details.get("full_generic_name"),
            "strength": details.get("strength"),
            "route": details.get("route"),
            "rxterms_dose_form": details.get("rxterms_dose_form"),
            "rxnorm_dose_form": details.get("rxnorm_dose_form"),
            "term_type": details.get("term_type"),
            "generic_rxcui": details.get("generic_rxcui"),
            "primary_ingredient_rxcui": details.get("primary_ingredient_rxcui"),
            "primary_ingredient_name": details.get("primary_ingredient_name"),
            "ingredients_json": json.dumps(details.get("ingredients", [])),
            "source": details.get("source"),
        },
    )


def get_medication_details(rxcui: str) -> dict:
    rxcui = str(rxcui or "").strip()
    if not rxcui:
        raise MedicationManagementError("get_medication_details failed: rxcui is required.")

    cached = _detail_memory_cache.get(rxcui)
    if cached and cached["expires_at"] > time.time():
        result = dict(cached["value"])
        result["cache"] = "memory"
        return result

    try:
        durable = _load_durable_cache(rxcui)
    except SqlClientError as exc:
        raise _wrap(exc, "get_medication_details") from exc
    if durable is not None:
        _detail_memory_cache[rxcui] = {"value": durable, "expires_at": time.time() + _DETAIL_CACHE_TTL_SECONDS}
        return durable

    try:
        details = rxnorm_client.fetch_medication_details(rxcui)
    except RxNormError as exc:
        raise _wrap(exc, "get_medication_details") from exc
    details["cache"] = "miss"

    try:
        _save_durable_cache(details)
    except SqlClientError:
        # Caching is best-effort; a write failure must not break the lookup.
        pass

    _detail_memory_cache[rxcui] = {"value": details, "expires_at": time.time() + _DETAIL_CACHE_TTL_SECONDS}
    return details


# ============================================================
# PATIENT MEDICATIONS (list / add / remove)
# ============================================================


def list_patient_medications(patient_id: str) -> dict:
    patient_id = (patient_id or "").strip()
    if not patient_id:
        raise MedicationManagementError("list_patient_medications failed: patient_id is required.")

    try:
        rows = execute_sql(
            f"""
            SELECT medication_record_id, patient_id, selected_rxcui, selected_name,
                   primary_ingredient_rxcui, primary_ingredient_name, ingredients_json,
                   strength, route, dose_form, term_type, medication_type, medication_status,
                   source, added_at, updated_at
            FROM {_TABLE_PATIENT_MEDICATIONS}
            WHERE patient_id = :patient_id AND is_active = true
            ORDER BY added_at
            """,
            parameters={"patient_id": patient_id},
            expect_rows=True,
        )
    except SqlClientError as exc:
        raise _wrap(exc, "list_patient_medications") from exc

    medications = []
    for row in rows:
        item = dict(row)
        try:
            item["ingredients"] = json.loads(item.get("ingredients_json") or "[]")
        except (TypeError, ValueError):
            item["ingredients"] = []
        item.pop("ingredients_json", None)
        medications.append(item)

    return {"patient_id": patient_id, "medications": medications, "count": len(medications)}


def _persist_auto_discovered_rule(finding: dict) -> None:
    """Mirrors the retired medication-lookup-api's _persist_auto_discovered_rule.

    Note: unlike that endpoint (which queried drug_interactions live), this
    rule becomes visible to medication-safety-agent's static rule matching
    only on its NEXT redeploy — its rule snapshot is baked in at deploy
    time, not queried live. See docs/databricks-endpoint-lifecycle.md.
    """
    evidence_id = finding.get("evidence_id")

    existing = execute_sql(
        f"SELECT evidence_id FROM {_TABLE_DRUG_INTERACTIONS} WHERE evidence_id = :evidence_id LIMIT 1",
        parameters={"evidence_id": evidence_id},
        expect_rows=True,
    )
    if existing:
        return

    execute_sql(
        f"""
        INSERT INTO {_TABLE_SOURCE_EVIDENCE}
        (evidence_id, drug, drug_rxcui, interaction_class, source, source_setid,
         section_code, section_title, subsection_title, evidence_text, reviewed, created_at)
        VALUES
        (:evidence_id, :drug, :drug_rxcui, :interaction_class, :source, :source_setid,
         :section_code, :section_title, :subsection_title, :evidence_text, false, current_timestamp())
        """,
        parameters={
            "evidence_id": evidence_id,
            "drug": finding.get("labeled_drug"),
            "drug_rxcui": finding.get("labeled_drug_rxcui"),
            "interaction_class": "Auto-discovered (openFDA)",
            "source": "openFDA",
            "source_setid": finding.get("trusted_source_set_id"),
            "section_code": finding.get("trusted_source_section_code"),
            "section_title": finding.get("trusted_source_field"),
            "subsection_title": None,
            "evidence_text": finding.get("trusted_source_excerpt"),
        },
    )

    execute_sql(
        f"""
        INSERT INTO {_TABLE_DRUG_INTERACTIONS}
        (drug_a, drug_b, interaction, severity, drug_a_rxcui, drug_b_rxcui, evidence_id, source, review_status)
        VALUES
        (:drug_a, :drug_b, :interaction, :severity, :drug_a_rxcui, :drug_b_rxcui, :evidence_id, :source, :review_status)
        """,
        parameters={
            "drug_a": finding.get("drug_a"),
            "drug_b": finding.get("drug_b"),
            "interaction": finding.get("concern"),
            "severity": "Unreviewed",
            "drug_a_rxcui": finding.get("new_medication_rxcui"),
            "drug_b_rxcui": finding.get("existing_medication_rxcui"),
            "evidence_id": evidence_id,
            "source": "openFDA",
            "review_status": "Auto-discovered, unreviewed",
        },
    )


def _current_medications_context(patient_id: str) -> list[dict]:
    listing = list_patient_medications(patient_id)
    return [
        {"selected_name": m.get("selected_name"), "ingredients": m.get("ingredients") or []}
        for m in listing["medications"]
    ]


def pre_add_check(patient_id: str, rxcui: str, labs: dict | None) -> dict:
    """Public pre_add_check: gathers what medication-safety-agent needs
    (candidate details + current medications, both owned locally now) and
    calls the centrally-versioned clinical check. Reconstructs patient_id/
    candidate_medication in the response so the JSON shape matches what the
    frontend already expects from before this migration.
    """
    details = get_medication_details(rxcui)
    current_medications = _current_medications_context(patient_id)

    context = {
        "candidate_name": details.get("selected_name"),
        "candidate_ingredients": details.get("ingredients") or [],
        "current_medications": current_medications,
    }

    try:
        result = check_pre_add(candidate_rxcui=rxcui, context=context, labs=labs or None)
    except DatabricksClientError as exc:
        raise _wrap(exc, "pre_add_check") from exc

    result["patient_id"] = patient_id
    result["candidate_medication"] = details
    return result


def add_patient_medication(patient_id: str, rxcui: str, medication_status: str, medication_type: str, labs: dict | None) -> dict:
    patient_id = (patient_id or "").strip()
    rxcui = str(rxcui or "").strip()
    if not patient_id:
        raise MedicationManagementError("add_patient_medication failed: patient_id is required.")
    if not rxcui:
        raise MedicationManagementError("add_patient_medication failed: rxcui is required.")

    medication_status = (medication_status or "CURRENT").upper()
    if medication_status not in {"CURRENT", "PROPOSED"}:
        raise MedicationManagementError("medication_status must be CURRENT or PROPOSED.")
    medication_type = (medication_type or "UNKNOWN").upper()
    if medication_type not in {"RX", "OTC", "UNKNOWN"}:
        raise MedicationManagementError("medication_type must be RX, OTC, or UNKNOWN.")

    details = get_medication_details(rxcui)

    # ---- Duplicate check ----
    try:
        existing = execute_sql(
            f"""
            SELECT medication_record_id, selected_name, medication_status
            FROM {_TABLE_PATIENT_MEDICATIONS}
            WHERE patient_id = :patient_id AND selected_rxcui = :selected_rxcui AND is_active = true
            LIMIT 1
            """,
            parameters={"patient_id": patient_id, "selected_rxcui": rxcui},
            expect_rows=True,
        )
    except SqlClientError as exc:
        raise _wrap(exc, "add_patient_medication") from exc

    if existing:
        return {
            "added": False,
            "duplicate": True,
            "patient_id": patient_id,
            "selected_rxcui": rxcui,
            "existing_record": existing[0],
            "message": "This medication is already active on the patient list.",
        }

    # ---- Safety check BEFORE persistence (unchanged ordering) ----
    safety = pre_add_check(patient_id=patient_id, rxcui=rxcui, labs=labs)

    for finding in safety.get("drug_drug_findings", []):
        if finding.get("review_status") == "Auto-discovered, unreviewed":
            try:
                _persist_auto_discovered_rule(finding)
            except SqlClientError:
                # Persisting the learned rule is best-effort; must not block the add.
                pass

    medication_record_id = str(uuid.uuid4())
    dose_form = details.get("rxnorm_dose_form") or details.get("rxterms_dose_form")

    try:
        execute_sql(
            f"""
            INSERT INTO {_TABLE_PATIENT_MEDICATIONS}
            (medication_record_id, patient_id, selected_rxcui, selected_name,
             primary_ingredient_rxcui, primary_ingredient_name, ingredients_json,
             strength, route, dose_form, term_type, medication_type, medication_status,
             source, is_active, added_at, updated_at)
            VALUES
            (:medication_record_id, :patient_id, :selected_rxcui, :selected_name,
             :primary_ingredient_rxcui, :primary_ingredient_name, :ingredients_json,
             :strength, :route, :dose_form, :term_type, :medication_type, :medication_status,
             :source, true, current_timestamp(), current_timestamp())
            """,
            parameters={
                "medication_record_id": medication_record_id,
                "patient_id": patient_id,
                "selected_rxcui": rxcui,
                "selected_name": details.get("selected_name"),
                "primary_ingredient_rxcui": details.get("primary_ingredient_rxcui"),
                "primary_ingredient_name": details.get("primary_ingredient_name"),
                "ingredients_json": json.dumps(details.get("ingredients", [])),
                "strength": details.get("strength"),
                "route": details.get("route"),
                "dose_form": dose_form,
                "term_type": details.get("term_type"),
                "medication_type": medication_type,
                "medication_status": medication_status,
                "source": details.get("source"),
            },
        )
    except SqlClientError as exc:
        raise _wrap(exc, "add_patient_medication") from exc

    if safety.get("safety_review_required") and medication_status == "PROPOSED":
        user_message = (
            "Medication saved as PROPOSED. Safety findings were identified; review them with the "
            "prescribing clinician or pharmacist before starting the medication."
        )
    elif safety.get("safety_review_required"):
        user_message = (
            "Medication recorded as CURRENT. Safety findings were identified. The medication remains "
            "on the patient list so the record reflects what the patient is taking."
        )
    else:
        user_message = (
            "Medication added. No supported safety concern was identified from the current prototype "
            "rules and available patient data."
        )

    return {
        "added": True,
        "duplicate": False,
        "medication_record_id": medication_record_id,
        "patient_id": patient_id,
        "medication": details,
        "medication_type": medication_type,
        "medication_status": medication_status,
        "safety": safety,
        "message": user_message,
    }


def remove_patient_medication(patient_id: str, medication_record_id: str) -> dict:
    patient_id = (patient_id or "").strip()
    medication_record_id = str(medication_record_id or "").strip()
    if not patient_id:
        raise MedicationManagementError("remove_patient_medication failed: patient_id is required.")
    if not medication_record_id:
        raise MedicationManagementError("remove_patient_medication failed: medication_record_id is required.")

    try:
        execute_sql(
            f"""
            UPDATE {_TABLE_PATIENT_MEDICATIONS}
            SET is_active = false, medication_status = 'DISCONTINUED', updated_at = current_timestamp()
            WHERE patient_id = :patient_id AND medication_record_id = :medication_record_id AND is_active = true
            """,
            parameters={"patient_id": patient_id, "medication_record_id": medication_record_id},
        )
    except SqlClientError as exc:
        raise _wrap(exc, "remove_patient_medication") from exc

    return {
        "removed": True,
        "patient_id": patient_id,
        "medication_record_id": medication_record_id,
        "message": "Medication marked DISCONTINUED and retained for history.",
    }
