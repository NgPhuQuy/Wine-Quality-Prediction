from fastapi import APIRouter
from ..services.modelInfoService import get_model_metadata  # FIX: import đúng function

router = APIRouter()

@router.get("/model/metadata")
async def fetch_model_metadata():
    return get_model_metadata()  # FIX: gọi đúng function thay vì gọi module "model_info()"
