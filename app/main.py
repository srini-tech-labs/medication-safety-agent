from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import medications, patients
from app.config import settings
from app.services.medication_management import MedicationManagementError
from app.web.routes import router as web_router

app = FastAPI(title="Medication Safety Agent")


@app.exception_handler(MedicationManagementError)
async def medication_management_error_handler(request: Request, exc: MedicationManagementError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

app.include_router(web_router)
app.include_router(patients.router)
app.include_router(medications.router)
