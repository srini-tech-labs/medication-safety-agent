from pydantic import BaseModel


class Labs(BaseModel):
    # The medication-safety-agent model's signature was inferred from its logging
    # example and locked to exactly these two columns/types — see
    # docs/databricks-deployment-status.md. Not a general lab dict. Shared with
    # medication-lookup-api's pre_add_check, which takes the same shape.
    egfr: int
    potassium: float
