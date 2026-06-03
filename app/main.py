from fastapi import FastAPI, HTTPException
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


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_array = np.array([request.features])

    if request.model_type == "tensorflow" and loader.tf_model:
        pred = loader.tf_model.predict(input_array)
        result = float(pred[0][0])

        return PredictionResponse(
            prediction=result,
            model_type="tensorflow",
            status="success"
        )

    elif request.model_type == "pytorch" and loader.pt_model:
        import torch

        with torch.no_grad():
            tensor = torch.tensor(input_array, dtype=torch.float32)
            pred = loader.pt_model(tensor)
            result = float(pred.numpy()[0])

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