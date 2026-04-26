from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..schemas.wine import WineCreate, WineResponse
from ..services import predictService  
from ..core.database import get_db
from ..routers import predict
from ..models.user import User

router = APIRouter(prefix="/predict", tags=["Prediction"])

# CHỈ GIỮ LẠI MỘT HÀM POST DUY NHẤT: Vừa dự đoán, vừa lưu
@router.post("/", response_model=WineResponse)
def predict_and_save(
    payload: WineCreate, 
    db: Session = Depends(get_db)
    # BƯỚC 1: XÓA DÒNG current_user: User = Depends(get_current_user)
):
    # BƯỚC 2: Lấy user_id trực tiếp từ payload thay vì từ current_user
    result = predictService.predict_and_store_wine(db, payload, payload.user_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    # BƯỚC 3: Trả về kết quả (Vẫn giữ logic cũ nhưng dùng payload.model_dump)
    return {
        **payload.model_dump(), 
        "id": result["id"],
        "quality_score": result["quality_score"], # Dùng điểm thật từ AI trả về
        "created_at": result["created_at"]
    }
# API LẤY LỊCH SỬ
@router.get("/history", response_model=list[WineResponse])
def get_user_history(
    db: Session = Depends(get_db), 
   
):
    
    return predictService.get_history_by_user(db, user_id)