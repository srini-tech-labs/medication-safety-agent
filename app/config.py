from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"))

    static_dir: Path = BASE_DIR / "app" / "static"
    templates_dir: Path = BASE_DIR / "app" / "web" / "templates"

    # Matches the OAuth profile already created via `databricks auth login`
    # and the deployed endpoint name — see docs/databricks-deployment-status.md.
    databricks_profile: str = "medication-safety-agent"
    serving_endpoint_name: str = "medication-safety-agent"

    # Used by the local Medication Management implementation to query Unity
    # Catalog directly (patient_medications, rxnorm_medication_cache) via the
    # SQL Statement Execution API — see docs/databricks-endpoint-lifecycle.md
    # for why this moved out of a Model Serving endpoint.
    databricks_warehouse_id: str = "0000000000000000"
    unity_catalog: str = "healthcare"
    unity_schema: str = "medication_safety"

    rxnorm_min_search_characters: int = 3
    rxnorm_default_search_limit: int = 10

    # When true, the two Databricks-backed calls (medication-safety-agent
    # analyze/pre_add_check, and Unity Catalog SQL) are replaced with
    # offline canned fixtures — see app/services/demo_fixtures.py and
    # app/services/demo_store.py. Lets a fresh clone demo the UI with no
    # Databricks workspace, profile, or credentials. RxNorm search/details
    # stay live either way (a public API, no credentials required).
    demo_mode: bool = False


settings = Settings()
