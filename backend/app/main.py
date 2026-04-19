from fastapi import FastAPI
from backend.app.schemas.wine import WineInput 
from .services.predictService import predict_wine
from .services.modelInfoService import get_model_metadata
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Wine AI API running"}

@app.get("/model-info")
def get_info():
    return get_model_metadata()

@app.post("/predict")
def predict(data: WineInput):
    return predict_wine(data)