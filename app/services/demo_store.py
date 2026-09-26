"""In-memory stand-in for Unity Catalog SQL, used only when DEMO_MODE is on.

This does NOT reimplement Medication Management's business logic — every
caller in app/services/medication_management.py runs completely unchanged
against this module; only the SQL execution boundary (sql_client.execute_sql)
is swapped out. Duplicate detection, caching, and persistence ordering are
all exercised for real, just against an ephemeral in-process store instead
of a real SQL warehouse. State resets whenever the process restarts.
"""

from datetime import UTC, datetime

_patient_medications: list[dict] = []
_medication_cache: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def reset() -> None:
    """Test helper: clear all in-memory demo state."""
    _patient_medications.clear()
    _medication_cache.clear()


def demo_execute_sql(statement: str, parameters: dict | None = None, expect_rows: bool = False) -> list[dict]:
    parameters = parameters or {}
    normalized = " ".join(statement.split())
    verb = normalized.strip().upper().split(" ", 1)[0]

    if "sql_status" in normalized:
        return [{"sql_status": "OK"}]

    if "rxnorm_medication_cache" in normalized:
        if verb == "SELECT":
            row = _medication_cache.get(parameters.get("rxcui"))
            return [row] if row else []
        if verb == "MERGE":
            rxcui = parameters.get("selected_rxcui")
            _medication_cache[rxcui] = {
                "selected_rxcui": rxcui,
                "selected_name": parameters.get("selected_name"),
                "full_name": parameters.get("full_name"),
                "full_generic_name": parameters.get("full_generic_name"),
                "strength": parameters.get("strength"),
                "route": parameters.get("route"),
                "rxterms_dose_form": parameters.get("rxterms_dose_form"),
                "rxnorm_dose_form": parameters.get("rxnorm_dose_form"),
                "term_type": parameters.get("term_type"),
                "generic_rxcui": parameters.get("generic_rxcui"),
                "primary_ingredient_rxcui": parameters.get("primary_ingredient_rxcui"),
                "primary_ingredient_name": parameters.get("primary_ingredient_name"),
                "ingredients_json": parameters.get("ingredients_json"),
                "source": parameters.get("source"),
            }
            return []

    if "drug_interactions" in normalized or "source_evidence" in normalized:
        # Demo mode's pre_add_check is a fixed canned rule table (see
        # demo_fixtures.py) — it never reports an "Auto-discovered,
        # unreviewed" finding, so this path is not exercised in practice.
        # Kept as a harmless no-op so the real code above it never breaks.
        if verb == "SELECT":
            return []
        return []

    if "patient_medications" in normalized:
        if verb == "SELECT" and "ORDER BY added_at" in normalized:
            patient_id = parameters.get("patient_id")
            return [
                {k: v for k, v in row.items() if k != "is_active"}
                for row in _patient_medications
                if row["patient_id"] == patient_id and row["is_active"]
            ]
        if verb == "SELECT":
            patient_id = parameters.get("patient_id")
            selected_rxcui = parameters.get("selected_rxcui")
            for row in _patient_medications:
                if row["patient_id"] == patient_id and row["selected_rxcui"] == selected_rxcui and row["is_active"]:
                    return [{
                        "medication_record_id": row["medication_record_id"],
                        "selected_name": row["selected_name"],
                        "medication_status": row["medication_status"],
                    }]
            return []
        if verb == "INSERT":
            _patient_medications.append({
                "medication_record_id": parameters.get("medication_record_id"),
                "patient_id": parameters.get("patient_id"),
                "selected_rxcui": parameters.get("selected_rxcui"),
                "selected_name": parameters.get("selected_name"),
                "primary_ingredient_rxcui": parameters.get("primary_ingredient_rxcui"),
                "primary_ingredient_name": parameters.get("primary_ingredient_name"),
                "ingredients_json": parameters.get("ingredients_json"),
                "strength": parameters.get("strength"),
                "route": parameters.get("route"),
                "dose_form": parameters.get("dose_form"),
                "term_type": parameters.get("term_type"),
                "medication_type": parameters.get("medication_type"),
                "medication_status": parameters.get("medication_status"),
                "source": parameters.get("source"),
                "added_at": _now(),
                "updated_at": _now(),
                "is_active": True,
            })
            return []
        if verb == "UPDATE":
            patient_id = parameters.get("patient_id")
            medication_record_id = parameters.get("medication_record_id")
            for row in _patient_medications:
                if row["patient_id"] == patient_id and row["medication_record_id"] == medication_record_id and row["is_active"]:
                    row["is_active"] = False
                    row["medication_status"] = "DISCONTINUED"
                    row["updated_at"] = _now()
            return []

    return []
