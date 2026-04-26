from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..schemas.wine import WineCreate, WineResponse
from ..services import predictService
from ..core.database import get_db
# FIX: Bỏ "from ..routers import predict" - đây là self-import gây circular import
# FIX: Bỏ "from ..models.user import User" - không dùng

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=WineResponse)
def predict_wine(  # FIX: đổi tên từ "predict" thành "predict_wine" để không đụng tên với module
    payload: WineCreate,
    db: Session = Depends(get_db)
):
    result = predictService.predict_and_store_wine(db, payload, payload.user_id)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return {
        **payload.model_dump(),
        "id": result["id"],
        "quality_score": result["quality_score"],
        "created_at": result["created_at"]
    }


@router.get("/history", response_model=list[WineResponse])
def get_user_history(  # FIX: thêm thụt lề đúng (trước đây body nằm ngoài hàm)
    user_id: int,
    db: Session = Depends(get_db)
):
    return predictService.get_history_by_user(db, user_id)
