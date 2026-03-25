# from fastapi import FastAPI
# from app.schemas.wine import WineInput 
# from app.services.model import predict_wine
# from fastapi.middleware.cors import CORSMiddleware


# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def root():
#     return {"message": "Wine AI API running"}

# @app.post("/predict")
# def predict(data: WineInput):
#     return predict_wine(data)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.predict import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Wine AI API running"}