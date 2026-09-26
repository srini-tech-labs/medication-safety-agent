# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # ⚠️ RETIRED — historical snapshot, not part of the active runtime architecture
# MAGIC
# MAGIC This notebook is kept for historical reference only. It is **not deployed** and does not
# MAGIC back any currently active Model Serving endpoint. The `medication-lookup-api` endpoint it
# MAGIC describes has been retired — its deterministic CRUD/lookup responsibilities now run locally
# MAGIC in `app/services/medication_management.py`, and its one genuinely clinical operation (the
# MAGIC pre-add safety check) was folded into the `medication-safety-agent` endpoint instead. See
# MAGIC `../docs/databricks-endpoint-lifecycle.md` for the full explanation of why, and
# MAGIC `../databricks/README.md` for what is currently active.

# COMMAND ----------

# MAGIC %md
# MAGIC # Medication Lookup API — MLflow PyFunc Serving Endpoint
# MAGIC
# MAGIC This notebook builds a Databricks-hosted backend for:
# MAGIC
# MAGIC - RxNorm type-ahead medication search
# MAGIC - RxTerms medication enrichment
# MAGIC - Ingredient RxCUI normalization
# MAGIC - Current + proposed medication persistence
# MAGIC - OTC/Rx status field support
# MAGIC - Pre-add safety screening against the current prototype rules
# MAGIC - Soft-delete/discontinue medication history
# MAGIC - Deployment as an MLflow `pyfunc` Unity Catalog model
# MAGIC - Exposure through a Databricks Model Serving endpoint
# MAGIC
# MAGIC **Synthetic/fictional patient data only. Not for clinical use.**
# MAGIC
# MAGIC The UI stays thin. It sends `action + payload_json` to this endpoint and renders the returned JSON.

# COMMAND ----------

# ============================================================
# CONFIGURATION
# CHANGE VALUES ONLY IN THIS CELL WHEN NEEDED.
# ============================================================

CATALOG = "healthcare"
SCHEMA = "medication_safety"

# Registered Unity Catalog model name.
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.medication_lookup_pyfunc"

# Databricks Model Serving endpoint name.
SERVING_ENDPOINT_NAME = "medication-lookup-api"

# Leave blank to automatically use the first available SQL warehouse.
# If you want a specific warehouse, paste only its warehouse ID here.
SQL_WAREHOUSE_ID = ""

# RxNorm autocomplete settings.
MIN_SEARCH_CHARACTERS = 3
DEFAULT_SEARCH_LIMIT = 10

# In-memory cache for type-ahead results.
SEARCH_CACHE_TTL_SECONDS = 60 * 60          # 1 hour

# Selected-medication detail cache.
DETAIL_CACHE_TTL_SECONDS = 60 * 60 * 24     # 24 hours
DURABLE_DETAIL_CACHE_DAYS = 1               # Delta-table cache

# Databricks SQL execution settings used inside serving.
SQL_WAIT_TIMEOUT = "10s"
SQL_POLL_TIMEOUT_SECONDS = 30

# Dynamic DailyMed/FDA fallback signal (used only when the prototype
# drug_interactions table has no rule for a given pair). This LLM endpoint is
# used ONLY to explain an already-retrieved FDA label excerpt in plain
# language — it never determines whether an interaction exists or how severe
# it is. That fact comes solely from the presence of the excerpt itself.
LLM_EXPLAIN_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
OPENFDA_TIMEOUT_SECONDS = 10

# Model Serving compute settings.
ENDPOINT_WORKLOAD_SIZE = "Small"
ENDPOINT_SCALE_TO_ZERO = True

print("Configuration loaded.")

# COMMAND ----------

# ============================================================
# IMPORTS + RESOLVE WORKSPACE / SQL WAREHOUSE
# NORMALLY NO CHANGES NEEDED IN THIS CELL.
# ============================================================

import json
import os
import time
import uuid

import mlflow
import mlflow.pyfunc
import pandas as pd
import requests

from mlflow.models import infer_signature
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksSQLWarehouse, DatabricksTable
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole,
    EndpointCoreConfigInput,
    ServedEntityInput,
)

workspace = WorkspaceClient()

WORKSPACE_HOST = workspace.config.host.rstrip("/")

if SQL_WAREHOUSE_ID:
    RESOLVED_WAREHOUSE_ID = SQL_WAREHOUSE_ID
else:
    warehouses = list(workspace.warehouses.list())

    if not warehouses:
        raise RuntimeError(
            "No SQL warehouse found. Create a serverless SQL warehouse "
            "or set SQL_WAREHOUSE_ID in the CONFIGURATION cell."
        )

    running = [
        wh
        for wh in warehouses
        if str(getattr(wh, "state", "")).upper().endswith("RUNNING")
    ]

    selected_warehouse = running[0] if running else warehouses[0]
    RESOLVED_WAREHOUSE_ID = selected_warehouse.id

print("Workspace:", WORKSPACE_HOST)
print("SQL Warehouse ID:", RESOLVED_WAREHOUSE_ID)
print("MLflow version:", mlflow.__version__)

# COMMAND ----------

# ============================================================
# CREATE PERSISTENT DELTA TABLES
# CHANGE ONLY THIS CELL IF THE PERSISTENT DATA MODEL CHANGES.
# ============================================================

# Patient medication history.
spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.patient_medications (
        medication_record_id STRING,
        patient_id STRING,

        selected_rxcui STRING,
        selected_name STRING,

        primary_ingredient_rxcui STRING,
        primary_ingredient_name STRING,
        ingredients_json STRING,

        strength STRING,
        route STRING,
        dose_form STRING,
        term_type STRING,

        medication_type STRING,
        medication_status STRING,

        source STRING,

        is_active BOOLEAN,

        added_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    USING DELTA
    """
)

# Durable cache for medication details AFTER a medication is selected.
# We intentionally do not persist every autocomplete keystroke.
spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.rxnorm_medication_cache (
        selected_rxcui STRING,
        selected_name STRING,

        full_name STRING,
        full_generic_name STRING,

        strength STRING,
        route STRING,
        rxterms_dose_form STRING,
        rxnorm_dose_form STRING,
        term_type STRING,
        generic_rxcui STRING,

        primary_ingredient_rxcui STRING,
        primary_ingredient_name STRING,
        ingredients_json STRING,

        source STRING,

        retrieved_at TIMESTAMP,
        expires_at TIMESTAMP
    )
    USING DELTA
    """
)

print("Persistent medication tables are ready.")

# COMMAND ----------

# ============================================================
# MLFLOW PYFUNC BACKEND
#
# Each ACTION has its own clearly marked section.
# When behavior changes later, change only that ACTION section.
# ============================================================

class MedicationLookupPyFunc(mlflow.pyfunc.PythonModel):
    """
    Action-based backend for medication lookup and patient medication management.

    PUBLIC INPUT COLUMNS
    --------------------
    action       : STRING
    payload_json : STRING containing a JSON object

    PUBLIC OUTPUT COLUMN
    --------------------
    result_json  : STRING containing a JSON response envelope
    """

    # ========================================================
    # ACTION: INITIALIZATION
    # Change only when model-wide behavior changes.
    # ========================================================

    def __init__(
        self,
        catalog,
        schema,
        min_search_characters=3,
        default_search_limit=10,
        search_cache_ttl_seconds=3600,
        detail_cache_ttl_seconds=86400,
        durable_detail_cache_days=1,
        sql_wait_timeout="10s",
        sql_poll_timeout_seconds=30,
    ):
        self.catalog = catalog
        self.schema = schema

        self.min_search_characters = int(min_search_characters)
        self.default_search_limit = int(default_search_limit)

        self.search_cache_ttl_seconds = int(search_cache_ttl_seconds)
        self.detail_cache_ttl_seconds = int(detail_cache_ttl_seconds)
        self.durable_detail_cache_days = int(durable_detail_cache_days)

        self.sql_wait_timeout = sql_wait_timeout
        self.sql_poll_timeout_seconds = int(sql_poll_timeout_seconds)

        self.patient_medications_table = (
            f"{catalog}.{schema}.patient_medications"
        )

        self.medication_cache_table = (
            f"{catalog}.{schema}.rxnorm_medication_cache"
        )

        # These are today's prototype rule tables.
        # Later we will replace the PRE-ADD SAFETY CHECK section
        # with the dynamic DailyMed/FDA evidence engine.
        self.drug_interactions_table = (
            f"{catalog}.{schema}.drug_interactions"
        )

        self.drug_lab_rules_table = (
            f"{catalog}.{schema}.drug_lab_rules"
        )

        self.source_evidence_table = (
            f"{catalog}.{schema}.source_evidence"
        )

    def load_context(self, context):
        # Match the existing Medication Safety Agent: MLflow resource
        # declarations provide automatic Databricks authentication at serving
        # time. No PAT, secret scope, or token environment variable is needed.
        self.workspace = WorkspaceClient()
        self.databricks_host = self.workspace.config.host.rstrip("/")
        self.sql_warehouse_id = context.model_config["sql_warehouse_id"]

        # NLM HTTP session.
        self.http = requests.Session()
        self.http.headers.update(
            {"User-Agent": "MedicationSafetyPrototype/1.0"}
        )

        # Databricks SQL Statement Execution API session.
        self.databricks_http = requests.Session()
        self.databricks_http.headers.update(
            {
                **self.workspace.config.authenticate(),
                "Content-Type": "application/json",
            }
        )

        # Fast caches inside the serving process.
        self.search_cache = {}
        self.detail_cache = {}

        # Existing prototype rules cached for five minutes.
        self.rule_cache = {
            "loaded_at": 0,
            "drug_interactions": [],
            "drug_lab_rules": [],
        }

    # ========================================================
    # ACTION: CACHE HELPERS
    # ========================================================

    def _cache_get(self, cache, key):
        item = cache.get(key)

        if not item:
            return None

        if time.time() >= item["expires_at"]:
            cache.pop(key, None)
            return None

        return item["value"]

    def _cache_put(self, cache, key, value, ttl_seconds):
        cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl_seconds,
        }

    # ========================================================
    # ACTION: RXNORM / RXTERMS HTTP ACCESS
    # Change only if NLM endpoints change.
    # ========================================================

    def _rxnav_get(self, path, params=None, timeout=15):
        url = "https://rxnav.nlm.nih.gov" + path

        response = self.http.get(
            url,
            params=params or {},
            timeout=timeout,
        )

        response.raise_for_status()
        return response.json()

    # ========================================================
    # ACTION: DATABRICKS SQL ACCESS
    #
    # The model-serving container does not use notebook Spark.
    # It reads/writes the Delta tables through the Databricks
    # SQL Statement Execution API.
    # ========================================================

    def _execute_sql(
        self,
        statement,
        parameters=None,
        expect_rows=False,
    ):
        url = (
            f"{self.databricks_host}"
            "/api/2.0/sql/statements"
        )

        body = {
            "warehouse_id": self.sql_warehouse_id,
            "statement": statement,
            "wait_timeout": self.sql_wait_timeout,
            "on_wait_timeout": "CONTINUE",
            "format": "JSON_ARRAY",
            "disposition": "INLINE",
            "parameters": parameters or [],
        }

        response = self.databricks_http.post(
            url,
            json=body,
            timeout=30,
        )

        response.raise_for_status()
        result = response.json()

        deadline = (
            time.time()
            + self.sql_poll_timeout_seconds
        )

        while (
            result.get("status", {}).get("state")
            not in {
                "SUCCEEDED",
                "FAILED",
                "CANCELED",
                "CLOSED",
            }
        ):
            if time.time() >= deadline:
                raise TimeoutError(
                    "Databricks SQL statement did not complete in time."
                )

            statement_id = result["statement_id"]

            time.sleep(0.5)

            poll = self.databricks_http.get(
                (
                    f"{self.databricks_host}"
                    "/api/2.0/sql/statements/"
                    f"{statement_id}"
                ),
                timeout=15,
            )

            poll.raise_for_status()
            result = poll.json()

        state = result.get(
            "status", {}
        ).get("state")

        if state != "SUCCEEDED":
            error = result.get(
                "status", {}
            ).get("error", {})

            raise RuntimeError(
                "Databricks SQL failed: "
                + json.dumps(error)
            )

        if not expect_rows:
            return []

        columns = [
            column["name"]
            for column in (
                result.get("manifest", {})
                .get("schema", {})
                .get("columns", [])
            )
        ]

        data_array = (
            result.get("result", {})
            .get("data_array", [])
            or []
        )

        return [
            dict(zip(columns, row))
            for row in data_array
        ]

    # ========================================================
    # ACTION: MEDICATION SEARCH / AUTOCOMPLETE
    #
    # 1. Requires 3+ characters.
    # 2. Searches Prescribable RxNorm first.
    # 3. Falls back to general RxNorm for broader coverage,
    #    including medication concepts that may be OTC.
    # 4. De-duplicates by RxCUI.
    # 5. Caches frequent typeahead searches in memory.
    # ========================================================

    def search_medications(
        self,
        query,
        limit=None,
    ):
        query = (query or "").strip()

        if (
            len(query)
            < self.min_search_characters
        ):
            return {
                "query": query,
                "minimum_characters":
                    self.min_search_characters,
                "matches": [],
                "message":
                    (
                        "Enter at least "
                        f"{self.min_search_characters} "
                        "characters."
                    ),
            }

        limit = int(
            limit
            or self.default_search_limit
        )

        limit = max(
            1,
            min(limit, 25),
        )

        cache_key = (
            query.lower(),
            limit,
        )

        cached = self._cache_get(
            self.search_cache,
            cache_key,
        )

        if cached is not None:
            result = dict(cached)
            result["cache"] = "memory"
            return result

        matches_by_rxcui = {}

        # ----------------------------------------------------
        # Search Prescribable RxNorm first.
        # ----------------------------------------------------

        prescribable = self._rxnav_get(
            (
                "/REST/Prescribe/"
                "approximateTerm.json"
            ),
            params={
                "term": query,
                "maxEntries":
                    min(limit * 3, 100),
                "option": 1,
            },
        )

        candidates = (
            prescribable
            .get("approximateGroup", {})
            .get("candidate", [])
            or []
        )

        for candidate in candidates:
            rxcui = candidate.get("rxcui")
            name = candidate.get("name")

            if not rxcui or not name:
                continue

            matches_by_rxcui[
                rxcui
            ] = {
                "rxcui": rxcui,
                "name": name,
                "rank":
                    int(
                        candidate.get(
                            "rank", 9999
                        )
                    ),
                "score":
                    float(
                        candidate.get(
                            "score", 0
                        )
                    ),
                "search_scope":
                    "PRESCRIBABLE_RXNORM",
            }

        # ----------------------------------------------------
        # General RxNorm fallback.
        # ----------------------------------------------------

        if len(matches_by_rxcui) < limit:
            general = self._rxnav_get(
                "/REST/approximateTerm.json",
                params={
                    "term": query,
                    "maxEntries":
                        min(limit * 3, 100),
                    "option": 1,
                },
            )

            candidates = (
                general
                .get("approximateGroup", {})
                .get("candidate", [])
                or []
            )

            for candidate in candidates:
                rxcui = candidate.get(
                    "rxcui"
                )

                name = candidate.get(
                    "name"
                )

                if (
                    not rxcui
                    or not name
                    or rxcui
                    in matches_by_rxcui
                ):
                    continue

                matches_by_rxcui[
                    rxcui
                ] = {
                    "rxcui": rxcui,
                    "name": name,
                    "rank":
                        int(
                            candidate.get(
                                "rank", 9999
                            )
                        ),
                    "score":
                        float(
                            candidate.get(
                                "score", 0
                            )
                        ),
                    "search_scope":
                        "RXNORM_FALLBACK",
                }

        priority = {
            "PRESCRIBABLE_RXNORM": 0,
            "RXNORM_FALLBACK": 1,
        }

        matches = sorted(
            matches_by_rxcui.values(),
            key=lambda item: (
                priority.get(
                    item["search_scope"],
                    9,
                ),
                item["rank"],
                -item["score"],
                item["name"].lower(),
            ),
        )[:limit]

        result = {
            "query": query,
            "matches": matches,
            "count": len(matches),
            "cache": "miss",
            "source":
                "NLM RxNorm / Prescribable RxNorm",
        }

        self._cache_put(
            self.search_cache,
            cache_key,
            result,
            self.search_cache_ttl_seconds,
        )

        return result

    # ========================================================
    # ACTION: DURABLE SELECTED-MEDICATION CACHE
    #
    # Typeahead remains in-memory.
    # Only selected/enriched medications are stored in Delta.
    # ========================================================

    def _load_durable_medication_cache(
        self,
        rxcui,
    ):
        rows = self._execute_sql(
            f"""
            SELECT
                selected_rxcui,
                selected_name,
                full_name,
                full_generic_name,
                strength,
                route,
                rxterms_dose_form,
                rxnorm_dose_form,
                term_type,
                generic_rxcui,
                primary_ingredient_rxcui,
                primary_ingredient_name,
                ingredients_json,
                source
            FROM {self.medication_cache_table}
            WHERE selected_rxcui = :rxcui
              AND expires_at > current_timestamp()
            ORDER BY retrieved_at DESC
            LIMIT 1
            """,
            parameters=[
                {
                    "name": "rxcui",
                    "value": str(rxcui),
                }
            ],
            expect_rows=True,
        )

        if not rows:
            return None

        row = rows[0]

        try:
            ingredients = json.loads(
                row.get("ingredients_json")
                or "[]"
            )
        except Exception:
            ingredients = []

        return {
            "selected_rxcui":
                row.get("selected_rxcui"),
            "selected_name":
                row.get("selected_name"),
            "full_name":
                row.get("full_name"),
            "full_generic_name":
                row.get("full_generic_name"),
            "strength":
                row.get("strength"),
            "route":
                row.get("route"),
            "rxterms_dose_form":
                row.get("rxterms_dose_form"),
            "rxnorm_dose_form":
                row.get("rxnorm_dose_form"),
            "term_type":
                row.get("term_type"),
            "generic_rxcui":
                row.get("generic_rxcui"),
            "primary_ingredient_rxcui":
                row.get(
                    "primary_ingredient_rxcui"
                ),
            "primary_ingredient_name":
                row.get(
                    "primary_ingredient_name"
                ),
            "ingredients":
                ingredients,
            "medication_type":
                "UNKNOWN",
            "source":
                row.get("source"),
            "cache":
                "delta",
        }

    def _save_durable_medication_cache(
        self,
        details,
    ):
        parameters = [
            {
                "name": "selected_rxcui",
                "value":
                    details.get(
                        "selected_rxcui"
                    ),
            },
            {
                "name": "selected_name",
                "value":
                    details.get(
                        "selected_name"
                    ),
            },
            {
                "name": "full_name",
                "value":
                    details.get(
                        "full_name"
                    ),
            },
            {
                "name": "full_generic_name",
                "value":
                    details.get(
                        "full_generic_name"
                    ),
            },
            {
                "name": "strength",
                "value":
                    details.get(
                        "strength"
                    ),
            },
            {
                "name": "route",
                "value":
                    details.get(
                        "route"
                    ),
            },
            {
                "name": "rxterms_dose_form",
                "value":
                    details.get(
                        "rxterms_dose_form"
                    ),
            },
            {
                "name": "rxnorm_dose_form",
                "value":
                    details.get(
                        "rxnorm_dose_form"
                    ),
            },
            {
                "name": "term_type",
                "value":
                    details.get(
                        "term_type"
                    ),
            },
            {
                "name": "generic_rxcui",
                "value":
                    details.get(
                        "generic_rxcui"
                    ),
            },
            {
                "name":
                    "primary_ingredient_rxcui",
                "value":
                    details.get(
                        "primary_ingredient_rxcui"
                    ),
            },
            {
                "name":
                    "primary_ingredient_name",
                "value":
                    details.get(
                        "primary_ingredient_name"
                    ),
            },
            {
                "name": "ingredients_json",
                "value":
                    json.dumps(
                        details.get(
                            "ingredients",
                            [],
                        )
                    ),
            },
            {
                "name": "source",
                "value":
                    details.get(
                        "source"
                    ),
            },
        ]

        self._execute_sql(
            f"""
            MERGE INTO
                {self.medication_cache_table}
                AS target

            USING (
                SELECT
                    :selected_rxcui
                    AS selected_rxcui
            )
            AS source

            ON
                target.selected_rxcui
                =
                source.selected_rxcui

            WHEN MATCHED THEN
            UPDATE SET
                selected_name =
                    :selected_name,
                full_name =
                    :full_name,
                full_generic_name =
                    :full_generic_name,
                strength =
                    :strength,
                route =
                    :route,
                rxterms_dose_form =
                    :rxterms_dose_form,
                rxnorm_dose_form =
                    :rxnorm_dose_form,
                term_type =
                    :term_type,
                generic_rxcui =
                    :generic_rxcui,
                primary_ingredient_rxcui =
                    :primary_ingredient_rxcui,
                primary_ingredient_name =
                    :primary_ingredient_name,
                ingredients_json =
                    :ingredients_json,
                source =
                    :source,
                retrieved_at =
                    current_timestamp(),
                expires_at =
                    current_timestamp()
                    + INTERVAL
                      {self.durable_detail_cache_days}
                      DAYS

            WHEN NOT MATCHED THEN
            INSERT (
                selected_rxcui,
                selected_name,
                full_name,
                full_generic_name,
                strength,
                route,
                rxterms_dose_form,
                rxnorm_dose_form,
                term_type,
                generic_rxcui,
                primary_ingredient_rxcui,
                primary_ingredient_name,
                ingredients_json,
                source,
                retrieved_at,
                expires_at
            )
            VALUES (
                :selected_rxcui,
                :selected_name,
                :full_name,
                :full_generic_name,
                :strength,
                :route,
                :rxterms_dose_form,
                :rxnorm_dose_form,
                :term_type,
                :generic_rxcui,
                :primary_ingredient_rxcui,
                :primary_ingredient_name,
                :ingredients_json,
                :source,
                current_timestamp(),
                current_timestamp()
                + INTERVAL
                  {self.durable_detail_cache_days}
                  DAYS
            )
            """,
            parameters=parameters,
        )

    # ========================================================
    # ACTION: SELECTED MEDICATION DETAILS / RXTERMS ENRICHMENT
    #
    # RxTerms gives product-friendly information.
    # RxNorm history/relationships resolve ingredient RxCUIs.
    # ========================================================

    def get_medication_details(
        self,
        rxcui,
    ):
        rxcui = str(
            rxcui or ""
        ).strip()

        if not rxcui:
            raise ValueError(
                "rxcui is required."
            )

        # Fast memory cache first.
        cached = self._cache_get(
            self.detail_cache,
            rxcui,
        )

        if cached is not None:
            result = dict(cached)
            result["cache"] = "memory"
            return result

        # Durable Delta cache second.
        durable = (
            self
            ._load_durable_medication_cache(
                rxcui
            )
        )

        if durable is not None:
            self._cache_put(
                self.detail_cache,
                rxcui,
                durable,
                self.detail_cache_ttl_seconds,
            )
            return durable

        # ----------------------------------------------------
        # RxNorm properties.
        # ----------------------------------------------------

        properties_json = self._rxnav_get(
            (
                f"/REST/rxcui/"
                f"{rxcui}/properties.json"
            )
        )

        properties = (
            properties_json.get(
                "properties"
            )
            or {}
        )

        rxnorm_name = (
            properties.get("name")
            or ""
        )

        rxnorm_tty = (
            properties.get("tty")
            or ""
        )

        # ----------------------------------------------------
        # RxTerms enrichment.
        # ----------------------------------------------------

        rxterms_json = self._rxnav_get(
            (
                "/REST/RxTerms/"
                f"rxcui/{rxcui}/"
                "allinfo.json"
            )
        )

        rxterms = (
            rxterms_json.get(
                "rxtermsProperties"
            )
            or {}
        )

        # ----------------------------------------------------
        # Resolve ingredients from RxNorm history status.
        # ----------------------------------------------------

        history_json = self._rxnav_get(
            (
                f"/REST/rxcui/"
                f"{rxcui}/"
                "historystatus.json"
            )
        )

        derived = (
            history_json
            .get("rxcuiStatusHistory", {})
            .get("derivedConcepts", {})
            or {}
        )

        ingredient_concepts = (
            derived.get(
                "ingredientConcept"
            )
            or []
        )

        ingredients = []

        for ingredient in ingredient_concepts:
            ingredient_rxcui = (
                ingredient.get(
                    "ingredientRxcui"
                )
            )

            ingredient_name = (
                ingredient.get(
                    "ingredientName"
                )
            )

            if (
                ingredient_rxcui
                and ingredient_name
            ):
                ingredients.append(
                    {
                        "rxcui":
                            str(
                                ingredient_rxcui
                            ),
                        "name":
                            ingredient_name,
                    }
                )

        # Ingredient itself was selected.
        if (
            not ingredients
            and rxnorm_tty
            in {"IN", "PIN"}
        ):
            ingredients = [
                {
                    "rxcui":
                        rxcui,
                    "name":
                        rxnorm_name,
                }
            ]

        # Final fallback: related IN concepts.
        if not ingredients:
            related_json = self._rxnav_get(
                (
                    f"/REST/rxcui/"
                    f"{rxcui}/related.json"
                ),
                params={
                    "tty": "IN"
                },
            )

            groups = (
                related_json
                .get("relatedGroup", {})
                .get("conceptGroup", [])
                or []
            )

            for group in groups:
                concepts = (
                    group.get(
                        "conceptProperties"
                    )
                    or []
                )

                for concept in concepts:
                    if (
                        concept.get("rxcui")
                        and concept.get("name")
                    ):
                        ingredients.append(
                            {
                                "rxcui":
                                    str(
                                        concept[
                                            "rxcui"
                                        ]
                                    ),
                                "name":
                                    concept[
                                        "name"
                                    ],
                            }
                        )

        # De-duplicate ingredients by RxCUI.
        ingredients = list(
            {
                item["rxcui"]: item
                for item in ingredients
            }.values()
        )

        primary = (
            ingredients[0]
            if ingredients
            else {}
        )

        full_name = (
            rxterms.get("fullName")
            or rxnorm_name
        )

        details = {
            "selected_rxcui":
                rxcui,

            "selected_name":
                full_name,

            "rxnorm_name":
                rxnorm_name,

            "full_name":
                full_name,

            "full_generic_name":
                rxterms.get(
                    "fullGenericName"
                ),

            "strength":
                rxterms.get(
                    "strength"
                ),

            "route":
                rxterms.get(
                    "route"
                ),

            "rxterms_dose_form":
                rxterms.get(
                    "rxtermsDoseForm"
                ),

            "rxnorm_dose_form":
                rxterms.get(
                    "rxnormDoseForm"
                ),

            "term_type":
                (
                    rxterms.get(
                        "termType"
                    )
                    or rxnorm_tty
                ),

            "generic_rxcui":
                rxterms.get(
                    "genericRxcui"
                ),

            "primary_ingredient_rxcui":
                primary.get(
                    "rxcui"
                ),

            "primary_ingredient_name":
                primary.get(
                    "name"
                ),

            "ingredients":
                ingredients,

            # RxNorm/RxTerms alone do not reliably tell us
            # whether the selected product is OTC or Rx-only.
            # DailyMed/openFDA will enrich this later.
            "medication_type":
                "UNKNOWN",

            "source":
                "NLM RxNorm + RxTerms",

            "cache":
                "miss",
        }

        # Save selected medication detail to Delta.
        self._save_durable_medication_cache(
            details
        )

        # Also save to fast memory cache.
        self._cache_put(
            self.detail_cache,
            rxcui,
            details,
            self.detail_cache_ttl_seconds,
        )

        return details

    # ========================================================
    # ACTION: LOAD CURRENT PROTOTYPE RULES
    #
    # Later replace only this section + PRE-ADD SAFETY CHECK
    # with dynamic DailyMed/FDA evidence retrieval.
    # ========================================================

    def _load_prototype_rules(self):
        # Refresh every five minutes.
        if (
            time.time()
            - self.rule_cache["loaded_at"]
            < 300
        ):
            return (
                self.rule_cache[
                    "drug_interactions"
                ],
                self.rule_cache[
                    "drug_lab_rules"
                ],
            )

        drug_interactions = (
            self._execute_sql(
                f"""
                SELECT *
                FROM
                    {self.drug_interactions_table}
                """,
                expect_rows=True,
            )
        )

        drug_lab_rules = (
            self._execute_sql(
                f"""
                SELECT *
                FROM
                    {self.drug_lab_rules_table}
                """,
                expect_rows=True,
            )
        )

        self.rule_cache = {
            "loaded_at":
                time.time(),

            "drug_interactions":
                drug_interactions,

            "drug_lab_rules":
                drug_lab_rules,
        }

        return (
            drug_interactions,
            drug_lab_rules,
        )

    # ========================================================
    # ACTION: DYNAMIC DAILYMED/FDA FALLBACK SIGNAL
    #
    # Used ONLY when the prototype drug_interactions table has no
    # rule for a given ingredient pair. This never lets a model
    # decide whether an interaction exists or how severe it is —
    # that fact is the FDA label excerpt itself, retrieved
    # verbatim. The LLM is used only to explain that excerpt in
    # plain language, exactly like the AI-explanation step in the
    # main Medication Safety Agent explains already-established
    # findings rather than inventing them.
    # ========================================================

    _OPENFDA_SECTION_LOINC = {
        "drug_interactions": "34073-7",
        "warnings_and_cautions": "43685-7",
        "contraindications": "34070-3",
    }

    def _openfda_label(self, generic_name):
        generic_name = (generic_name or "").strip()

        if not generic_name:
            return None

        try:
            response = self.http.get(
                "https://api.fda.gov/drug/label.json",
                params={
                    "search": f'openfda.generic_name:"{generic_name.upper()}"',
                    "limit": 1,
                },
                timeout=OPENFDA_TIMEOUT_SECONDS,
            )

            if response.status_code != 200:
                return None

            results = response.json().get("results") or []

            return results[0] if results else None

        except Exception:
            # openFDA is a best-effort fallback signal, not a
            # required dependency — a lookup failure here must not
            # break pre_add_check, it just means no dynamic signal
            # is available this time.
            return None

    def _openfda_mention(self, label, other_name):
        other_name = (other_name or "").strip().lower()

        if not other_name or not label:
            return None

        for field in (
            "drug_interactions",
            "warnings_and_cautions",
            "contraindications",
        ):
            values = label.get(field)

            if not values:
                continue

            text = (
                " ".join(values)
                if isinstance(values, list)
                else str(values)
            )

            idx = text.lower().find(other_name)

            if idx == -1:
                continue

            start = max(0, idx - 200)
            end = min(len(text), idx + len(other_name) + 400)

            return {
                "field": field,
                "section_code": self._OPENFDA_SECTION_LOINC.get(field, ""),
                "excerpt": text[start:end].strip(),
                "set_id": label.get("set_id"),
            }

        return None

    def _openfda_signal(
        self,
        candidate_name,
        candidate_rxcui,
        existing_name,
        existing_rxcui,
    ):
        # Interaction mentions in FDA labeling are not always
        # symmetric — check the candidate's own label first, then
        # fall back to checking the already-persisted drug's label
        # for a mention of the candidate.
        candidate_label = self._openfda_label(candidate_name)
        mention = (
            self._openfda_mention(candidate_label, existing_name)
            if candidate_label
            else None
        )

        if mention:
            return {
                **mention,
                "labeled_drug": candidate_name,
                "labeled_drug_rxcui": candidate_rxcui,
                "mentioned_drug": existing_name,
                "mentioned_drug_rxcui": existing_rxcui,
            }

        existing_label = self._openfda_label(existing_name)
        mention = (
            self._openfda_mention(existing_label, candidate_name)
            if existing_label
            else None
        )

        if mention:
            return {
                **mention,
                "labeled_drug": existing_name,
                "labeled_drug_rxcui": existing_rxcui,
                "mentioned_drug": candidate_name,
                "mentioned_drug_rxcui": candidate_rxcui,
            }

        return None

    def _llm_explain_excerpt(self, labeled_drug, mentioned_drug, excerpt):
        try:
            response = self.workspace.serving_endpoints.query(
                name=LLM_EXPLAIN_ENDPOINT,
                messages=[
                    ChatMessage(
                        role=ChatMessageRole.SYSTEM,
                        content=(
                            "You explain a single excerpt from an FDA drug "
                            "label in plain language for a patient. Only "
                            "explain what the excerpt itself says. Never "
                            "add a claim, a severity judgment, or a "
                            "recommendation that is not directly supported "
                            "by the excerpt's own wording. Keep it to 2-3 "
                            "short sentences."
                        ),
                    ),
                    ChatMessage(
                        role=ChatMessageRole.USER,
                        content=(
                            f"This excerpt is from the FDA label for "
                            f"{labeled_drug}, in the context of "
                            f"{mentioned_drug}:\n\n{excerpt}\n\n"
                            "Explain in plain language what this excerpt says."
                        ),
                    ),
                ],
                max_tokens=220,
            )

            return response.choices[0].message.content.strip()

        except Exception:
            # The raw excerpt (already surfaced separately) remains
            # available even if the explanation call fails.
            return None

    def _persist_auto_discovered_rule(self, finding):
        evidence_id = finding.get("evidence_id")

        # The in-memory rule cache (_load_prototype_rules) can stay stale on
        # a given served-model replica for up to 5 minutes, so a second add
        # for the same pair can re-trigger this fallback+persist path before
        # that replica sees the rule it (or another replica) already wrote.
        # evidence_id is deterministic per unordered rxcui pair, so check
        # for it first rather than inserting a duplicate curated-rule row.
        existing = self._execute_sql(
            f"""
            SELECT evidence_id
            FROM {self.drug_interactions_table}
            WHERE evidence_id = :evidence_id
            LIMIT 1
            """,
            parameters=[{"name": "evidence_id", "value": evidence_id}],
            expect_rows=True,
        )

        if existing:
            self.rule_cache["loaded_at"] = 0
            return

        self._execute_sql(
            f"""
            INSERT INTO {self.source_evidence_table}
            (
                evidence_id, drug, drug_rxcui, interaction_class,
                source, source_setid, section_code, section_title,
                subsection_title, evidence_text, reviewed, created_at
            )
            VALUES
            (
                :evidence_id, :drug, :drug_rxcui, :interaction_class,
                :source, :source_setid, :section_code, :section_title,
                :subsection_title, :evidence_text, false, current_timestamp()
            )
            """,
            parameters=[
                {"name": "evidence_id", "value": evidence_id},
                {"name": "drug", "value": finding.get("labeled_drug")},
                {"name": "drug_rxcui", "value": finding.get("labeled_drug_rxcui")},
                {"name": "interaction_class", "value": "Auto-discovered (openFDA)"},
                {"name": "source", "value": "openFDA"},
                {"name": "source_setid", "value": finding.get("trusted_source_set_id")},
                {"name": "section_code", "value": finding.get("trusted_source_section_code")},
                {"name": "section_title", "value": finding.get("trusted_source_field")},
                {"name": "subsection_title", "value": None},
                {"name": "evidence_text", "value": finding.get("trusted_source_excerpt")},
            ],
        )

        self._execute_sql(
            f"""
            INSERT INTO {self.drug_interactions_table}
            (drug_a, drug_b, interaction, severity, drug_a_rxcui, drug_b_rxcui, evidence_id, source, review_status)
            VALUES
            (:drug_a, :drug_b, :interaction, :severity, :drug_a_rxcui, :drug_b_rxcui, :evidence_id, :source, :review_status)
            """,
            parameters=[
                {"name": "drug_a", "value": finding.get("drug_a")},
                {"name": "drug_b", "value": finding.get("drug_b")},
                {"name": "interaction", "value": finding.get("concern")},
                {"name": "severity", "value": "Unreviewed"},
                {"name": "drug_a_rxcui", "value": finding.get("new_medication_rxcui")},
                {"name": "drug_b_rxcui", "value": finding.get("existing_medication_rxcui")},
                {"name": "evidence_id", "value": evidence_id},
                {"name": "source", "value": "openFDA"},
                {"name": "review_status", "value": "Auto-discovered, unreviewed"},
            ],
        )

        # Force the next _load_prototype_rules call in this (possibly
        # still-warm) container to see the newly persisted rule
        # immediately instead of waiting out the 5-minute cache TTL.
        self.rule_cache["loaded_at"] = 0

    # ========================================================
    # ACTION: LIST PATIENT MEDICATIONS
    # ========================================================

    def list_patient_medications(
        self,
        patient_id,
    ):
        patient_id = (
            patient_id or ""
        ).strip()

        if not patient_id:
            raise ValueError(
                "patient_id is required."
            )

        rows = self._execute_sql(
            f"""
            SELECT
                medication_record_id,
                patient_id,
                selected_rxcui,
                selected_name,
                primary_ingredient_rxcui,
                primary_ingredient_name,
                ingredients_json,
                strength,
                route,
                dose_form,
                term_type,
                medication_type,
                medication_status,
                source,
                added_at,
                updated_at
            FROM
                {self.patient_medications_table}
            WHERE
                patient_id = :patient_id
                AND is_active = true
            ORDER BY
                added_at
            """,
            parameters=[
                {
                    "name":
                        "patient_id",
                    "value":
                        patient_id,
                }
            ],
            expect_rows=True,
        )

        medications = []

        for row in rows:
            item = dict(row)

            try:
                item["ingredients"] = (
                    json.loads(
                        item.get(
                            "ingredients_json"
                        )
                        or "[]"
                    )
                )
            except Exception:
                item["ingredients"] = []

            item.pop(
                "ingredients_json",
                None,
            )

            medications.append(item)

        return {
            "patient_id":
                patient_id,
            "medications":
                medications,
            "count":
                len(medications),
        }

    # ========================================================
    # ACTION: PRE-ADD SAFETY CHECK
    #
    # TODAY:
    # Uses the existing prototype drug_interactions and
    # drug_lab_rules Delta tables.
    #
    # LATER:
    # Replace ONLY this section with dynamic trusted-source
    # retrieval from DailyMed/FDA + class matching.
    # ========================================================

    def pre_add_check(
        self,
        patient_id,
        selected_rxcui,
        labs=None,
    ):
        labs = labs or {}

        normalized_labs = {
            str(key).lower():
                value
            for key, value
            in labs.items()
        }

        candidate = (
            self.get_medication_details(
                selected_rxcui
            )
        )

        current = (
            self.list_patient_medications(
                patient_id
            )[
                "medications"
            ]
        )

        (
            ddi_rules,
            drug_lab_rules,
        ) = self._load_prototype_rules()

        new_ingredients = (
            candidate.get(
                "ingredients"
            )
            or []
        )

        current_ingredients = []

        for medication in current:
            ingredients = (
                medication.get(
                    "ingredients"
                )
                or []
            )

            for ingredient in ingredients:
                current_ingredients.append(
                    {
                        "selected_name":
                            medication.get(
                                "selected_name"
                            ),
                        "rxcui":
                            str(
                                ingredient.get(
                                    "rxcui"
                                )
                                or ""
                            ),
                        "name":
                            ingredient.get(
                                "name"
                            ),
                    }
                )

        drug_drug_findings = []

        # ----------------------------------------------------
        # Drug-Drug screening.
        # ----------------------------------------------------

        for new_ingredient in new_ingredients:
            new_rxcui = str(
                new_ingredient.get(
                    "rxcui"
                )
                or ""
            )

            for current_ingredient in (
                current_ingredients
            ):
                current_rxcui = str(
                    current_ingredient.get(
                        "rxcui"
                    )
                    or ""
                )

                for rule in ddi_rules:
                    rule_a = str(
                        rule.get(
                            "drug_a_rxcui"
                        )
                        or ""
                    )

                    rule_b = str(
                        rule.get(
                            "drug_b_rxcui"
                        )
                        or ""
                    )

                    match = (
                        (
                            new_rxcui
                            == rule_a
                            and
                            current_rxcui
                            == rule_b
                        )
                        or
                        (
                            new_rxcui
                            == rule_b
                            and
                            current_rxcui
                            == rule_a
                        )
                    )

                    if not match:
                        continue

                    drug_drug_findings.append(
                        {
                            "finding_type":
                                "Drug-Drug",

                            "severity":
                                rule.get(
                                    "severity"
                                ),

                            "new_medication":
                                candidate.get(
                                    "selected_name"
                                ),

                            "existing_medication":
                                current_ingredient.get(
                                    "selected_name"
                                ),

                            "drug_a":
                                rule.get(
                                    "drug_a"
                                ),

                            "drug_b":
                                rule.get(
                                    "drug_b"
                                ),

                            "concern":
                                rule.get(
                                    "interaction"
                                ),

                            "evidence_id":
                                rule.get(
                                    "evidence_id"
                                ),

                            "source":
                                rule.get(
                                    "source"
                                ),

                            "review_status":
                                rule.get(
                                    "review_status"
                                ),
                        }
                    )

        # ----------------------------------------------------
        # Dynamic DailyMed/FDA fallback — only attempted when the
        # prototype drug_interactions table found nothing for this
        # pair. The fact is the FDA label excerpt itself; the LLM
        # (below) only explains that excerpt, it never decides
        # whether the interaction exists.
        # ----------------------------------------------------

        if not drug_drug_findings:
            for new_ingredient in new_ingredients:
                if drug_drug_findings:
                    break

                for current_ingredient in current_ingredients:
                    new_rxcui_check = str(new_ingredient.get("rxcui") or "")
                    current_rxcui_check = str(current_ingredient.get("rxcui") or "")

                    # Same ingredient as one already on file (e.g. checking a
                    # different pack/formulation of a drug the patient already
                    # takes) — searching a drug's own label for its own name
                    # would always "match" itself, which is meaningless here.
                    if (
                        new_rxcui_check
                        and new_rxcui_check == current_rxcui_check
                    ):
                        continue

                    signal = self._openfda_signal(
                        candidate_name=new_ingredient.get("name"),
                        candidate_rxcui=new_rxcui_check,
                        existing_name=current_ingredient.get("name"),
                        existing_rxcui=current_rxcui_check,
                    )

                    if not signal:
                        continue

                    explanation = self._llm_explain_excerpt(
                        signal["labeled_drug"],
                        signal["mentioned_drug"],
                        signal["excerpt"],
                    )

                    new_rxcui = str(new_ingredient.get("rxcui") or "")
                    current_rxcui = str(current_ingredient.get("rxcui") or "")

                    drug_drug_findings.append(
                        {
                            "finding_type": "Drug-Drug",
                            "severity": "Unreviewed",
                            "new_medication": candidate.get("selected_name"),
                            "new_medication_rxcui": new_rxcui,
                            "existing_medication": current_ingredient.get("selected_name"),
                            "existing_medication_rxcui": current_rxcui,
                            "drug_a": new_ingredient.get("name") or candidate.get("selected_name"),
                            "drug_b": current_ingredient.get("name") or current_ingredient.get("selected_name"),
                            "concern": explanation or signal["excerpt"],
                            "trusted_source_excerpt": signal["excerpt"],
                            "trusted_source_field": signal["field"],
                            "trusted_source_section_code": signal["section_code"],
                            "trusted_source_set_id": signal["set_id"],
                            "labeled_drug": signal["labeled_drug"],
                            "ai_explanation": explanation,
                            "source": "openFDA",
                            "evidence_id": (
                                "FDA-" + "-".join(sorted([new_rxcui, current_rxcui]))
                            ),
                            "review_status": "Auto-discovered, unreviewed",
                        }
                    )

                    break

        drug_lab_findings = []

        # ----------------------------------------------------
        # Drug-Lab screening for the candidate medication.
        # ----------------------------------------------------

        for new_ingredient in new_ingredients:
            new_rxcui = str(
                new_ingredient.get(
                    "rxcui"
                )
                or ""
            )

            for rule in drug_lab_rules:
                if (
                    new_rxcui
                    != str(
                        rule.get(
                            "drug_rxcui"
                        )
                        or ""
                    )
                ):
                    continue

                lab_name = str(
                    rule.get(
                        "lab"
                    )
                    or ""
                ).lower()

                if (
                    lab_name
                    not in normalized_labs
                ):
                    continue

                try:
                    observed = float(
                        normalized_labs[
                            lab_name
                        ]
                    )

                    threshold = float(
                        rule.get(
                            "threshold"
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                operator = (
                    rule.get(
                        "operator"
                    )
                )

                triggered = (
                    (
                        operator == ">"
                        and observed
                        > threshold
                    )
                    or
                    (
                        operator == "<"
                        and observed
                        < threshold
                    )
                    or
                    (
                        operator == ">="
                        and observed
                        >= threshold
                    )
                    or
                    (
                        operator == "<="
                        and observed
                        <= threshold
                    )
                    or
                    (
                        operator == "="
                        and observed
                        == threshold
                    )
                )

                if not triggered:
                    continue

                drug_lab_findings.append(
                    {
                        "finding_type":
                            "Drug-Lab",

                        "severity":
                            rule.get(
                                "severity"
                            ),

                        "drug":
                            rule.get(
                                "drug"
                            ),

                        "lab":
                            rule.get(
                                "lab"
                            ),

                        "observed_value":
                            observed,

                        "operator":
                            operator,

                        "threshold":
                            threshold,

                        "concern":
                            rule.get(
                                "message"
                            ),

                        "threshold_basis":
                            rule.get(
                                "threshold_basis"
                            ),

                        "evidence_id":
                            rule.get(
                                "evidence_id"
                            ),

                        "source":
                            rule.get(
                                "source"
                            ),
                    }
                )

        severity_order = {
            "High": 1,
            "Moderate": 2,
            "Low": 3,
            "Unreviewed": 4,
        }

        drug_drug_findings.sort(
            key=lambda item:
                severity_order.get(
                    item.get(
                        "severity"
                    ),
                    99,
                )
        )

        drug_lab_findings.sort(
            key=lambda item:
                severity_order.get(
                    item.get(
                        "severity"
                    ),
                    99,
                )
        )

        all_findings = (
            drug_drug_findings
            + drug_lab_findings
        )

        safety_review_required = any(
            finding.get("severity")
            in {
                "High",
                "Moderate",
                "Unreviewed",
            }
            for finding
            in all_findings
        )

        has_unreviewed = any(
            finding.get("review_status") == "Auto-discovered, unreviewed"
            for finding in all_findings
        )

        if has_unreviewed:
            message = (
                "A potential interaction was found in FDA drug "
                "labeling that is not yet part of the reviewed "
                "prototype rules. Review the excerpt and "
                "explanation below and confirm with the "
                "prescribing clinician or pharmacist before "
                "starting a proposed medication. Existing "
                "medications should still be recorded accurately."
            )
        elif all_findings:
            message = (
                "Safety findings were identified "
                "from the current prototype rules. "
                "Review the findings before starting "
                "a proposed medication. Existing "
                "medications should still be recorded "
                "accurately."
            )
        else:
            message = (
                "No supported safety concern was "
                "identified from the current prototype "
                "rules and available patient data. "
                "This does not prove that the "
                "medication is risk-free."
            )

        return {
            "patient_id":
                patient_id,

            "candidate_medication":
                candidate,

            "drug_drug_findings":
                drug_drug_findings,

            "drug_lab_findings":
                drug_lab_findings,

            "total_findings":
                len(all_findings),

            "safety_review_required":
                safety_review_required,

            "message":
                message,

            "coverage_note":
                (
                    "Current pre-add screening uses "
                    "the prototype Databricks rule "
                    "tables. Dynamic DailyMed/FDA "
                    "evidence retrieval will replace "
                    "this section later."
                ),
        }

    # ========================================================
    # ACTION: ADD MEDICATION TO PATIENT
    #
    # IMPORTANT DESIGN:
    # - A safety finding does NOT prevent a CURRENT medication
    #   from being recorded.
    # - A medication being considered can be stored as PROPOSED.
    # - RX, OTC and UNKNOWN are all supported.
    # ========================================================

    def add_patient_medication(
        self,
        patient_id,
        selected_rxcui,
        medication_status="CURRENT",
        medication_type="UNKNOWN",
        labs=None,
    ):
        patient_id = (
            patient_id or ""
        ).strip()

        selected_rxcui = str(
            selected_rxcui or ""
        ).strip()

        if not patient_id:
            raise ValueError(
                "patient_id is required."
            )

        if not selected_rxcui:
            raise ValueError(
                "selected_rxcui is required."
            )

        medication_status = str(
            medication_status
            or "CURRENT"
        ).upper()

        if medication_status not in {
            "CURRENT",
            "PROPOSED",
        }:
            raise ValueError(
                "medication_status must be "
                "CURRENT or PROPOSED."
            )

        medication_type = str(
            medication_type
            or "UNKNOWN"
        ).upper()

        if medication_type not in {
            "RX",
            "OTC",
            "UNKNOWN",
        }:
            raise ValueError(
                "medication_type must be "
                "RX, OTC, or UNKNOWN."
            )

        details = (
            self.get_medication_details(
                selected_rxcui
            )
        )

        # ----------------------------------------------------
        # Duplicate check.
        # ----------------------------------------------------

        existing = self._execute_sql(
            f"""
            SELECT
                medication_record_id,
                selected_name,
                medication_status
            FROM
                {self.patient_medications_table}
            WHERE
                patient_id = :patient_id
                AND selected_rxcui =
                    :selected_rxcui
                AND is_active = true
            LIMIT 1
            """,
            parameters=[
                {
                    "name":
                        "patient_id",
                    "value":
                        patient_id,
                },
                {
                    "name":
                        "selected_rxcui",
                    "value":
                        selected_rxcui,
                },
            ],
            expect_rows=True,
        )

        if existing:
            return {
                "added":
                    False,

                "duplicate":
                    True,

                "patient_id":
                    patient_id,

                "selected_rxcui":
                    selected_rxcui,

                "existing_record":
                    existing[0],

                "message":
                    (
                        "This medication is already "
                        "active on the patient list."
                    ),
            }

        # ----------------------------------------------------
        # Run safety check BEFORE persistence.
        # ----------------------------------------------------

        safety = self.pre_add_check(
            patient_id=
                patient_id,

            selected_rxcui=
                selected_rxcui,

            labs=
                labs or {},
        )

        # ----------------------------------------------------
        # The user is proceeding with the add despite (or informed
        # by) any dynamic DailyMed/FDA fallback finding — persist
        # it into the prototype rule tables now, marked unreviewed,
        # so future checks for this pair use the fast static path
        # instead of repeating the live FDA lookup every time.
        # ----------------------------------------------------

        for finding in safety.get("drug_drug_findings", []):
            if finding.get("review_status") == "Auto-discovered, unreviewed":
                try:
                    self._persist_auto_discovered_rule(finding)
                except Exception:
                    # Persisting the learned rule is a best-effort
                    # enhancement — failing to persist it must not
                    # block the medication add itself.
                    pass

        medication_record_id = str(
            uuid.uuid4()
        )

        ingredients_json = json.dumps(
            details.get(
                "ingredients",
                [],
            )
        )

        dose_form = (
            details.get(
                "rxnorm_dose_form"
            )
            or details.get(
                "rxterms_dose_form"
            )
        )

        # ----------------------------------------------------
        # Persist medication.
        # ----------------------------------------------------

        self._execute_sql(
            f"""
            INSERT INTO
                {self.patient_medications_table}
            (
                medication_record_id,
                patient_id,

                selected_rxcui,
                selected_name,

                primary_ingredient_rxcui,
                primary_ingredient_name,
                ingredients_json,

                strength,
                route,
                dose_form,
                term_type,

                medication_type,
                medication_status,

                source,
                is_active,

                added_at,
                updated_at
            )
            VALUES
            (
                :medication_record_id,
                :patient_id,

                :selected_rxcui,
                :selected_name,

                :primary_ingredient_rxcui,
                :primary_ingredient_name,
                :ingredients_json,

                :strength,
                :route,
                :dose_form,
                :term_type,

                :medication_type,
                :medication_status,

                :source,
                true,

                current_timestamp(),
                current_timestamp()
            )
            """,
            parameters=[
                {
                    "name":
                        "medication_record_id",
                    "value":
                        medication_record_id,
                },
                {
                    "name":
                        "patient_id",
                    "value":
                        patient_id,
                },
                {
                    "name":
                        "selected_rxcui",
                    "value":
                        selected_rxcui,
                },
                {
                    "name":
                        "selected_name",
                    "value":
                        details.get(
                            "selected_name"
                        ),
                },
                {
                    "name":
                        "primary_ingredient_rxcui",
                    "value":
                        details.get(
                            "primary_ingredient_rxcui"
                        ),
                },
                {
                    "name":
                        "primary_ingredient_name",
                    "value":
                        details.get(
                            "primary_ingredient_name"
                        ),
                },
                {
                    "name":
                        "ingredients_json",
                    "value":
                        ingredients_json,
                },
                {
                    "name":
                        "strength",
                    "value":
                        details.get(
                            "strength"
                        ),
                },
                {
                    "name":
                        "route",
                    "value":
                        details.get(
                            "route"
                        ),
                },
                {
                    "name":
                        "dose_form",
                    "value":
                        dose_form,
                },
                {
                    "name":
                        "term_type",
                    "value":
                        details.get(
                            "term_type"
                        ),
                },
                {
                    "name":
                        "medication_type",
                    "value":
                        medication_type,
                },
                {
                    "name":
                        "medication_status",
                    "value":
                        medication_status,
                },
                {
                    "name":
                        "source",
                    "value":
                        details.get(
                            "source"
                        ),
                },
            ],
        )

        # ----------------------------------------------------
        # Build UI-friendly response message.
        # ----------------------------------------------------

        if (
            safety[
                "safety_review_required"
            ]
            and medication_status
            == "PROPOSED"
        ):
            user_message = (
                "Medication saved as PROPOSED. "
                "Safety findings were identified; "
                "review them with the prescribing "
                "clinician or pharmacist before "
                "starting the medication."
            )

        elif safety[
            "safety_review_required"
        ]:
            user_message = (
                "Medication recorded as CURRENT. "
                "Safety findings were identified. "
                "The medication remains on the "
                "patient list so the record reflects "
                "what the patient is taking."
            )

        else:
            user_message = (
                "Medication added. No supported "
                "safety concern was identified from "
                "the current prototype rules and "
                "available patient data."
            )

        return {
            "added":
                True,

            "duplicate":
                False,

            "medication_record_id":
                medication_record_id,

            "patient_id":
                patient_id,

            "medication":
                details,

            "medication_type":
                medication_type,

            "medication_status":
                medication_status,

            "safety":
                safety,

            "message":
                user_message,
        }

    # ========================================================
    # ACTION: REMOVE / DISCONTINUE MEDICATION
    #
    # Uses a soft delete to preserve medication history.
    # ========================================================

    def remove_patient_medication(
        self,
        patient_id,
        medication_record_id,
    ):
        patient_id = (
            patient_id or ""
        ).strip()

        medication_record_id = str(
            medication_record_id
            or ""
        ).strip()

        if not patient_id:
            raise ValueError(
                "patient_id is required."
            )

        if not medication_record_id:
            raise ValueError(
                "medication_record_id is required."
            )

        self._execute_sql(
            f"""
            UPDATE
                {self.patient_medications_table}
            SET
                is_active = false,
                medication_status =
                    'DISCONTINUED',
                updated_at =
                    current_timestamp()
            WHERE
                patient_id =
                    :patient_id
                AND medication_record_id =
                    :medication_record_id
                AND is_active = true
            """,
            parameters=[
                {
                    "name":
                        "patient_id",
                    "value":
                        patient_id,
                },
                {
                    "name":
                        "medication_record_id",
                    "value":
                        medication_record_id,
                },
            ],
        )

        return {
            "removed":
                True,

            "patient_id":
                patient_id,

            "medication_record_id":
                medication_record_id,

            "message":
                (
                    "Medication marked "
                    "DISCONTINUED and retained "
                    "for history."
                ),
        }

    # ========================================================
    # ACTION: HEALTH CHECK
    #
    # Tests:
    # - NLM RxNorm connectivity
    # - Databricks SQL connectivity
    # ========================================================

    def health_check(self):
        rxnorm_version = self._rxnav_get(
            "/REST/version.json"
        )

        sql_test = self._execute_sql(
            """
            SELECT
                'OK' AS sql_status
            """,
            expect_rows=True,
        )

        return {
            "status":
                "OK",

            "rxnorm":
                rxnorm_version,

            "databricks_sql":
                sql_test,

            "service":
                "medication-lookup-api",

            "code_marker":
                "review-status-passthrough-v2",
        }

    # ========================================================
    # ACTION: REQUEST ROUTER
    #
    # Add future API operations only in this section.
    # ========================================================

    def _dispatch(
        self,
        action,
        payload,
    ):
        action = str(
            action or ""
        ).strip().lower()

        if action == "health":
            return self.health_check()

        if action == "search_medications":
            return self.search_medications(
                query=
                    payload.get(
                        "query"
                    ),
                limit=
                    payload.get(
                        "limit"
                    ),
            )

        if action == "get_medication_details":
            return self.get_medication_details(
                rxcui=
                    payload.get(
                        "rxcui"
                    )
            )

        if action == "list_patient_medications":
            return self.list_patient_medications(
                patient_id=
                    payload.get(
                        "patient_id"
                    )
            )

        if action == "pre_add_check":
            return self.pre_add_check(
                patient_id=
                    payload.get(
                        "patient_id"
                    ),
                selected_rxcui=
                    payload.get(
                        "rxcui"
                    ),
                labs=
                    payload.get(
                        "labs"
                    )
                    or {},
            )

        if action == "add_patient_medication":
            return self.add_patient_medication(
                patient_id=
                    payload.get(
                        "patient_id"
                    ),
                selected_rxcui=
                    payload.get(
                        "rxcui"
                    ),
                medication_status=
                    payload.get(
                        "medication_status",
                        "CURRENT",
                    ),
                medication_type=
                    payload.get(
                        "medication_type",
                        "UNKNOWN",
                    ),
                labs=
                    payload.get(
                        "labs"
                    )
                    or {},
            )

        if action == "remove_patient_medication":
            return self.remove_patient_medication(
                patient_id=
                    payload.get(
                        "patient_id"
                    ),
                medication_record_id=
                    payload.get(
                        "medication_record_id"
                    ),
            )

        raise ValueError(
            "Unsupported action: "
            f"{action}"
        )

    # ========================================================
    # ACTION: MLFLOW PYFUNC PUBLIC CONTRACT
    #
    # Keep this stable so the UI remains thin.
    # ========================================================

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        if not isinstance(
            model_input,
            pd.DataFrame,
        ):
            model_input = pd.DataFrame(
                model_input
            )

        outputs = []

        for _, row in (
            model_input.iterrows()
        ):
            action = row.get(
                "action"
            )

            payload_json = row.get(
                "payload_json"
            )

            try:
                if (
                    payload_json is None
                    or str(
                        payload_json
                    ).strip() == ""
                ):
                    payload = {}

                elif isinstance(
                    payload_json,
                    dict,
                ):
                    payload = payload_json

                else:
                    payload = json.loads(
                        str(
                            payload_json
                        )
                    )

                result = self._dispatch(
                    action,
                    payload,
                )

                envelope = {
                    "ok":
                        True,
                    "action":
                        action,
                    "result":
                        result,
                    "error":
                        None,
                }

            except Exception as exc:
                envelope = {
                    "ok":
                        False,
                    "action":
                        action,
                    "result":
                        None,
                    "error":
                        {
                            "type":
                                type(
                                    exc
                                ).__name__,
                            "message":
                                str(exc),
                        },
                }

            outputs.append(
                {
                    "result_json":
                        json.dumps(
                            envelope,
                            default=str,
                        )
                }
            )

        return pd.DataFrame(
            outputs
        )

print("MedicationLookupPyFunc class loaded.")

# COMMAND ----------

# ============================================================
# LOG + REGISTER MLFLOW PYFUNC MODEL
# CHANGE ONLY THIS CELL IF THE MODEL REGISTRATION CHANGES.
# ============================================================

pyfunc_model = MedicationLookupPyFunc(
    catalog=CATALOG,
    schema=SCHEMA,

    min_search_characters=
        MIN_SEARCH_CHARACTERS,

    default_search_limit=
        DEFAULT_SEARCH_LIMIT,

    search_cache_ttl_seconds=
        SEARCH_CACHE_TTL_SECONDS,

    detail_cache_ttl_seconds=
        DETAIL_CACHE_TTL_SECONDS,

    durable_detail_cache_days=
        DURABLE_DETAIL_CACHE_DAYS,

    sql_wait_timeout=
        SQL_WAIT_TIMEOUT,

    sql_poll_timeout_seconds=
        SQL_POLL_TIMEOUT_SECONDS,
)

# Keep a very simple public request schema.
# Complex request data stays inside payload_json.
input_example = pd.DataFrame(
    [
        {
            "action":
                "search_medications",

            "payload_json":
                json.dumps(
                    {
                        "query": "lisi",
                        "limit": 5,
                    }
                ),
        }
    ]
)

output_example = pd.DataFrame(
    [
        {
            "result_json":
                json.dumps(
                    {
                        "ok": True,
                        "action":
                            "search_medications",
                        "result": {
                            "query": "lisi",
                            "matches": [],
                        },
                        "error": None,
                    }
                )
        }
    ]
)

signature = infer_signature(
    input_example,
    output_example,
)

# Declare every Databricks resource used by the served pyfunc. This follows
# the same automatic-authentication pattern as the existing Medication Safety
# Agent and avoids storing or injecting a personal access token.
MODEL_RESOURCES = [
    DatabricksSQLWarehouse(
        warehouse_id=RESOLVED_WAREHOUSE_ID,
    ),
    DatabricksTable(
        table_name=f"{CATALOG}.{SCHEMA}.drug_interactions",
    ),
    DatabricksTable(
        table_name=f"{CATALOG}.{SCHEMA}.drug_lab_rules",
    ),
    DatabricksTable(
        table_name=f"{CATALOG}.{SCHEMA}.rxnorm_medication_cache",
    ),
    DatabricksTable(
        table_name=f"{CATALOG}.{SCHEMA}.patient_medications",
    ),
    DatabricksTable(
        table_name=f"{CATALOG}.{SCHEMA}.source_evidence",
    ),
    DatabricksServingEndpoint(
        endpoint_name=LLM_EXPLAIN_ENDPOINT,
    ),
]

mlflow.set_tracking_uri(
    "databricks"
)

mlflow.set_registry_uri(
    "databricks-uc"
)

with mlflow.start_run(
    run_name=
        "medication-lookup-pyfunc"
):
    model_info = mlflow.pyfunc.log_model(
        name="model",

        python_model=
            pyfunc_model,

        signature=
            signature,

        input_example=
            input_example,

        model_config={
            "sql_warehouse_id":
                RESOLVED_WAREHOUSE_ID,
        },

        resources=
            MODEL_RESOURCES,

        # Pin MLflow to the version used to package this model.
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            "databricks-sdk",
            "pandas>=2.0,<3",
            "requests>=2.31,<3",
        ],
    )

registered_model = mlflow.register_model(
    model_uri=
        model_info.model_uri,

    name=
        UC_MODEL_NAME,
)

MODEL_VERSION = str(
    registered_model.version
)

print(
    "Registered model:",
    UC_MODEL_NAME,
)

print(
    "Registered version:",
    MODEL_VERSION,
)

# COMMAND ----------

# ============================================================
# CREATE OR UPDATE MODEL SERVING ENDPOINT
#
# CHANGE ONLY THIS CELL IF SERVING COMPUTE CHANGES.
# Authentication is supplied automatically from MODEL_RESOURCES.
# ============================================================

served_entity_name = (
    "medication-lookup-"
    + MODEL_VERSION
)

served_entities = [
    ServedEntityInput(
        name=served_entity_name,
        entity_name=UC_MODEL_NAME,
        entity_version=MODEL_VERSION,
        workload_size=ENDPOINT_WORKLOAD_SIZE,
        scale_to_zero_enabled=ENDPOINT_SCALE_TO_ZERO,
    )
]

# Determine whether the endpoint already exists.
endpoint_exists = False

try:
    workspace.serving_endpoints.get(
        name=
            SERVING_ENDPOINT_NAME
    )

    endpoint_exists = True

except Exception as exc:
    message = str(exc).lower()

    missing = any(
        text in message
        for text in [
            "not found",
            "does not exist",
            "resource_does_not_exist",
            "404",
        ]
    )

    if not missing:
        raise

if endpoint_exists:
    print(
        "Updating endpoint to model version",
        MODEL_VERSION,
    )

    workspace.serving_endpoints.update_config(
        name=
            SERVING_ENDPOINT_NAME,
        served_entities=
            served_entities,
    )

else:
    print(
        "Creating endpoint:",
        SERVING_ENDPOINT_NAME,
    )

    workspace.serving_endpoints.create(
        name=
            SERVING_ENDPOINT_NAME,
        config=
            EndpointCoreConfigInput(
                served_entities=served_entities,
            ),
    )

# Wait until deployment/update is complete.
deadline = (
    time.time()
    + (25 * 60)
)

while True:
    endpoint_object = (
        workspace.serving_endpoints.get(
            name=
                SERVING_ENDPOINT_NAME
        )
    )

    endpoint = endpoint_object.as_dict()

    state = endpoint.get(
        "state",
        {},
    )

    print(
        "Endpoint state:",
        json.dumps(
            state,
            default=str,
        ),
    )

    state_text = json.dumps(
        state,
        default=str,
    ).upper()

    if (
        '"READY"' in state_text
        and "NOT_READY"
        not in state_text
        and "IN_PROGRESS"
        not in state_text
        and "UPDATING"
        not in state_text
    ):
        break

    if time.time() >= deadline:
        raise TimeoutError(
            "Endpoint did not become READY "
            "within 25 minutes."
        )

    time.sleep(20)

print(
    "Serving endpoint is READY:",
    SERVING_ENDPOINT_NAME,
)

# COMMAND ----------

# ============================================================
# THIN CLIENT HELPER
#
# Your UI can use the SAME contract.
# It only needs to know:
#   action
#   payload
#
# The UI does NOT need RxNorm/RxTerms/business logic.
# ============================================================

def call_medication_lookup_endpoint(
    action,
    payload=None,
):
    response = workspace.serving_endpoints.query(
        name=
            SERVING_ENDPOINT_NAME,

        dataframe_split={
            "columns": [
                "action",
                "payload_json",
            ],
            "data": [
                [
                    action,
                    json.dumps(
                        payload or {}
                    ),
                ]
            ],
        },
    )

    if isinstance(
        response,
        dict,
    ):
        predictions = response.get(
            "predictions"
        )
    else:
        predictions = getattr(
            response,
            "predictions",
            None,
        )

    if not predictions:
        raise RuntimeError(
            "Endpoint returned no predictions: "
            + str(response)
        )

    first = predictions[0]

    if isinstance(
        first,
        dict,
    ):
        result_json = first.get(
            "result_json"
        )
    else:
        result_json = first

    return json.loads(
        result_json
    )

print("Thin endpoint client helper loaded.")

# COMMAND ----------

# ============================================================
# THIN CLIENT HELPER
#
# Your UI can use the SAME contract.
# It only needs to know:
#   action
#   payload
#
# The UI does NOT need RxNorm/RxTerms/business logic.
# ============================================================

from databricks.sdk.service.serving import DataframeSplitInput


def call_medication_lookup_endpoint(
    action,
    payload=None,
):
    response = workspace.serving_endpoints.query(
        name=SERVING_ENDPOINT_NAME,

        dataframe_split=DataframeSplitInput(
            columns=[
                "action",
                "payload_json",
            ],

            data=[
                [
                    action,
                    json.dumps(
                        payload or {}
                    ),
                ]
            ],
        ),
    )

    if isinstance(response, dict):
        predictions = response.get(
            "predictions"
        )
    else:
        predictions = getattr(
            response,
            "predictions",
            None,
        )

    if not predictions:
        raise RuntimeError(
            "Endpoint returned no predictions: "
            + str(response)
        )

    first_prediction = predictions[0]

    if isinstance(first_prediction, dict):
        result_json = first_prediction.get(
            "result_json"
        )
    else:
        result_json = first_prediction

    if not result_json:
        raise RuntimeError(
            "Endpoint prediction did not contain result_json: "
            + str(first_prediction)
        )

    return json.loads(
        result_json
    )


print("Thin endpoint client helper loaded.")