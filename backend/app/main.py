from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.database import engine, Base
from .routers import auth, predict
from .services.modelInfoService import get_model_metadata

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Wine Quality Prediction API - Team Phu Quy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # FIX: thiếu dấu đóng ngoặc )

# Đăng ký router
app.include_router(auth.router)
app.include_router(predict.router)

@app.get("/")
def root():
    """Trang chủ - kiểm tra trạng thái Backend"""
    return {"message": "Wine AI API is running smoothly!"}

@app.get("/model-info")
def get_info():
    """Lấy thông tin metadata các model AI"""
    return get_model_metadata()
