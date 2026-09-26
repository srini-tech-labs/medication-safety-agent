from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.schemas import Labs
from app.services import medication_management

router = APIRouter(prefix="/api/medications", tags=["medications"])


class PreAddCheckRequest(BaseModel):
    patient_id: str
    rxcui: str
    # Optional, not defaulted-empty-Labs: when the patient's labs genuinely
    # aren't available, never invent values or send zero-filled placeholders.
    labs: Labs | None = None


class AddMedicationRequest(BaseModel):
    patient_id: str
    rxcui: str
    medication_status: Literal["CURRENT", "PROPOSED"]
    medication_type: Literal["RX", "OTC", "UNKNOWN"]
    labs: Labs | None = None


class DiscontinueMedicationRequest(BaseModel):
    patient_id: str
    medication_record_id: str


@router.get("/health")
def health() -> dict:
    return medication_management.health()


@router.get("/patients/{patient_id}")
def list_patient_medications(patient_id: str) -> dict:
    return medication_management.list_patient_medications(patient_id)


@router.get("/search")
def search_medications(q: str = Query(min_length=3), limit: int = Query(10, ge=1, le=50)) -> dict:
    return medication_management.search_medications(q, limit)


@router.get("/rxcui/{rxcui}")
def get_medication_details(rxcui: str) -> dict:
    return medication_management.get_medication_details(rxcui)


@router.post("/pre-add-check")
def pre_add_check(request: PreAddCheckRequest) -> dict:
    labs = request.labs.model_dump() if request.labs else None
    return medication_management.pre_add_check(request.patient_id, request.rxcui, labs)


@router.post("/add")
def add_patient_medication(request: AddMedicationRequest) -> dict:
    labs = request.labs.model_dump() if request.labs else None
    return medication_management.add_patient_medication(
        request.patient_id, request.rxcui, request.medication_status, request.medication_type, labs
    )


@router.post("/discontinue")
def remove_patient_medication(request: DiscontinueMedicationRequest) -> dict:
    return medication_management.remove_patient_medication(request.patient_id, request.medication_record_id)
