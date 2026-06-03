import asyncio
from uuid import uuid4
from typing import Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from app.models import PredictionRequest, PredictionResponse
from model_loader import ModelLoader


app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = ModelLoader()


prediction_store: Dict[str, dict] = {}


async def process_prediction(request_id: str, data: dict):

    await asyncio.sleep(2)  # simulate heavy ML inference

    features = data["features"]
    model_type = data["model_type"]

    input_array = np.array([features])

    
    result = sum(features) / len(features)

    prediction_store[request_id]["status"] = "completed"
    prediction_store[request_id]["result"] = {
        "prediction": result,
        "model_type": model_type,
        "status": "success"
    }



@app.get("/health")
async def health():
    return {"status": "healthy"}



@app.post("/predict")
def predict(request: PredictionRequest):

    return {
        "message": "Use async endpoint /predict-async for processing"
    }



@app.post("/predict-async", status_code=202)
async def async_predict(request: PredictionRequest, background_tasks: BackgroundTasks):

    request_id = str(uuid4())

    prediction_store[request_id] = {
        "status": "processing",
        "result": None
    }

    background_tasks.add_task(
        process_prediction,
        request_id,
        request.dict()
    )

    return {
        "status": "processing",
        "request_id": request_id
    }



@app.get("/predict-result/{request_id}")
async def get_prediction_result(request_id: str):

    if request_id not in prediction_store:
        raise HTTPException(status_code=404, detail="Invalid request_id")

    record = prediction_store[request_id]

    return record