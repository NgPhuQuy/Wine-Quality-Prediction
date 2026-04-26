from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.wine import WineCreate, WineResponse
from app.services import predictService  
from app.core.database import get_db
from app.routers.auth import get_current_user 
from app.models.user import User

router = APIRouter(prefix="/predict", tags=["Prediction"])

# CHỈ GIỮ LẠI MỘT HÀM POST DUY NHẤT: Vừa dự đoán, vừa lưu
@router.post("/", response_model=WineResponse)
def predict_and_save(
    payload: WineCreate, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    # Gọi hàm "tất cả trong một" ở Service mà mình vừa sửa cho Quý
    result = predictService.predict_and_store_wine(db, payload, current_user.id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    # Trả về dữ liệu khớp với WineResponse (gồm các input + id + quality_score)
    return {
        **payload.model_dump(), # Lấy toàn bộ input của người dùng
        "id": result["id"],
        "quality_score": 7 if result["is_good_wine"] else 5, # Hoặc lấy result["quality_score"] nếu AI trả về điểm
        "created_at": result["created_at"]
    }

# API LẤY LỊCH SỬ
@router.get("/history", response_model=list[WineResponse])
def get_user_history(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    
    return predictService.get_history_by_user(db, current_user.id)