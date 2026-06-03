from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from model_loader import ModelLoader

app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready API for serving ML models",
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




class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)
    model_type: str = Field(default="tensorflow", pattern="^(tensorflow|pytorch)$")


class PredictionResponse(BaseModel):
    prediction: float
    model_type: str
    status: str


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
async def health_check():
    return {"status": "healthy"}



@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_array = np.array([request.features])

    
    if request.model_type == "tensorflow" and loader.tf_model:
        prediction = loader.tf_model.predict(input_array)
        result = float(prediction[0][0])

        return PredictionResponse(
            prediction=result,
            model_type="tensorflow",
            status="success"
        )

    
    elif request.model_type == "pytorch" and loader.pt_model:
        import torch

        with torch.no_grad():
            tensor_input = torch.tensor(input_array, dtype=torch.float32)
            prediction = loader.pt_model(tensor_input)
            result = float(prediction.numpy()[0])

        return PredictionResponse(
            prediction=result,
            model_type="pytorch",
            status="success"
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid model type or model not loaded"
        )




@app.post("/predict-async", response_model=PredictionResponse)
async def predict_async(request: PredictionRequest):

    input_array = np.array([request.features])

    if request.model_type == "tensorflow" and loader.tf_model:
        prediction = loader.tf_model.predict(input_array)
        result = float(prediction[0][0])

        return PredictionResponse(
            prediction=result,
            model_type="tensorflow",
            status="success"
        )

    elif request.model_type == "pytorch" and loader.pt_model:
        import torch

        with torch.no_grad():
            tensor_input = torch.tensor(input_array, dtype=torch.float32)
            prediction = loader.pt_model(tensor_input)
            result = float(prediction.numpy()[0])

        return PredictionResponse(
            prediction=result,
            model_type="pytorch",
            status="success"
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid model type or model not loaded"
        )