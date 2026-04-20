from fastapi import APIRouter
from . import model_info

router = APIRouter()

@router.get("/model/metadata")
async def fetch_model_metadata():
    return model_info()