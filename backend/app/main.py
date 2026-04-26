from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.predict import predict
from core.database import engine, Base
from routers import auth
from schemas.wine import WineCreate
from services.predictService import predict_wine
from services.modelInfoService import get_model_metadata

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Wine Quality Prediction API - Team Phu Quy")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. ĐĂNG KÝ ROUTER
# Đưa các API đăng ký/đăng nhập từ file auth vào hệ thống chính
app.include_router(auth.router)
app.include_router(predict.router)
# 6. ĐỊNH NGHĨA CÁC ENDPOINT (CÁC ĐƯỜNG DẪN)
@app.get("/")
def root():
    """Trang chủ kiểm tra trạng thái Backend"""
    return {"message": "Wine AI API is running smoothly!"}

@app.get("/model-info")
def get_info():
    """Lấy thông tin metadata của các model AI"""
    return get_model_metadata()

