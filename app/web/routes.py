from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory=str(settings.templates_dir))

# Seed data from docs/databricks-agent-notes.md. Ages aren't specified there
# (only meds/labs/expected risk), so they're placeholders for the demo form.
# labs uses the deployed model's fixed {egfr, potassium} shape, not the
# generic lab names ("K", "eGFR") the original notes described — see
# docs/databricks-deployment-status.md.
SEED_PATIENTS = {
    "P001": {
        "patient_id": "P001",
        "age": 68,
        "medications": ["Lisinopril", "Spironolactone", "Ibuprofen"],
        "conditions": [],
        "labs": {"potassium": 5.1, "egfr": 55},
        "allergies": [],
    },
    "P002": {
        "patient_id": "P002",
        "age": 55,
        "medications": ["Lisinopril", "Ibuprofen"],
        "conditions": [],
        "labs": {"potassium": 4.4, "egfr": 82},
        "allergies": [],
    },
    "P003": {
        "patient_id": "P003",
        "age": 72,
        "medications": ["Spironolactone", "Ibuprofen"],
        "conditions": [],
        "labs": {"potassium": 4.6, "egfr": 48},
        "allergies": [],
    },
}


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"seed_patients": SEED_PATIENTS, "demo_mode": settings.demo_mode}
    )
