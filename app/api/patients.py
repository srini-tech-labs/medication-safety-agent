from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import Labs
from app.services.databricks_client import DatabricksClientError, analyze_patient

router = APIRouter(prefix="/api/patients", tags=["patients"])


class PatientInput(BaseModel):
    patient_id: str
    age: int
    medications: list[str]
    conditions: list[str] = []
    labs: Labs
    allergies: list[str] = []


@router.post("/analyze")
def analyze(patient: PatientInput) -> dict:
    try:
        return analyze_patient(patient.model_dump())
    except DatabricksClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
