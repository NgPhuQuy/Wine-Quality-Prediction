from fastapi import APIRouter
from app.schemas.wine import WineInput
from app.services.model import predict_wine

router = APIRouter()

@router.post("/predict")
def predict(data: WineInput):
    result = predict_wine(data, model_name=data.model)

    return {
        "model": data.model,
        "quality": result["quality"],
        "time": result["time"]
    }

