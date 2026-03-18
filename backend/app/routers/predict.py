from fastapi import APIRouter
from schemas.wine import WineInput
from services.model import predict_wine

router = APIRouter(prefix="/predict", tags=["Predict"])

@router.post("/")
def predict_wine(data: WineInput):
    return predict_wine(data)