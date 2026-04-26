from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import IntEnum

# 1. Định nghĩa loại rượu cho rõ ràng
class WineType(IntEnum):
    RED = 0
    WHITE = 1

# 2. Schema cơ sở (Chứa các thông số hóa học)
class WineBase(BaseModel):
    fixed_acidity: float = Field(..., ge=0, json_schema_extra={"example": 7.4})
    volatile_acidity: float = Field(..., ge=0, json_schema_extra={"example": 0.7})
    citric_acid: float = Field(..., ge=0, json_schema_extra={"example": 0.0})
    residual_sugar: float = Field(..., ge=0, json_schema_extra={"example": 1.9})
    chlorides: float = Field(..., ge=0, json_schema_extra={"example": 0.076})
    free_sulfur_dioxide: float = Field(..., ge=0, json_schema_extra={"example": 11.0})
    total_sulfur_dioxide: float = Field(..., ge=0, json_schema_extra={"example": 34.0})
    density: float = Field(..., ge=0, json_schema_extra={"example": 0.9978})
    ph: float = Field(..., ge=0, le=14, json_schema_extra={"example": 3.51}) # Đổi pH thành ph
    sulphates: float = Field(..., ge=0, json_schema_extra={"example": 0.56})
    alcohol: float = Field(..., ge=0, json_schema_extra={"example": 9.4})
    wine_type: WineType = Field(default=WineType.RED, description="0: Red, 1: White")

# 3. Schema dùng khi Frontend gửi dữ liệu lên dự đoán
class WineCreate(WineBase):
    user_id: int

# 4. Schema dùng khi trả kết quả về (Có thêm ID và Điểm số từ AI)
class WineResponse(WineBase):
    id: int
    quality_score: int = Field(..., description="Dự đoán từ Model AI")
    created_at: datetime
    status: Optional[str] = None        # thêm
    quality_label: Optional[str] = None # thêm
    raw_score: Optional[float] = None 
    class Config:
        from_attributes = True # Quan trọng: Để biến đổi từ SQLAlchemy sang Pydantic