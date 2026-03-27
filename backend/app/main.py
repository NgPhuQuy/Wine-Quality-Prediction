from fastapi import FastAPI
from backend.app.schemas.wine import WineInput 
from backend.app.services.model import predict_wine
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

@app.post("/predict")
def predict(data: WineInput):
    return predict_wine(data)