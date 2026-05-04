from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..schemas.wine import WineCreate, WineResponse
from ..services import predictService
from ..core.database import get_db


router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=WineResponse)
def predict_wine( 
    payload: WineCreate,
    db: Session = Depends(get_db)
):
    result = predictService.predict_and_store_wine(db, payload, payload.user_id)
    print("=== RESULT ===", result) 
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return {
    **payload.model_dump(),
    "id": result["id"],
    "quality_score": result["quality_score"],
    "quality_label": result["quality_label"],
    "raw_score": result["raw_score"],
    "is_good_wine": result["is_good_wine"],
    "status": "success",
    "created_at": result["created_at"],
    }


@router.get("/history/{user_id}", response_model=list[WineResponse])
def get_user_history(
    user_id: int, 
    db: Session = Depends(get_db)
):
    history = predictService.get_history_by_user(db, user_id)
    return history