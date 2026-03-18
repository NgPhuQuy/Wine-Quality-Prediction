from fastapi import FastAPI
from backend.app.schemas.wine import WineInput 
from backend.app.services.model import predict_wine

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Wine AI API running"}

@app.post("/predict")
def predict(data: WineInput):
    return predict_wine(data)