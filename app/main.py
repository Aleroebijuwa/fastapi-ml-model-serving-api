from fastapi import (
    FastAPI,
    BackgroundTasks,
    HTTPException,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from datetime import datetime
import asyncio
from uuid import uuid4
from typing import Dict

from app.models import PredictionRequest, PredictionResponse
from model_loader import ModelLoader


# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================

class ModelLoadingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class PredictionError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    description="Production-ready API for serving machine learning models with TensorFlow and PyTorch support"
)


# ==========================================
# CUSTOM OPENAPI DOCUMENTATION
# ==========================================

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ML Model Serving API",
        version="1.0.0",
        description="Production-ready API for ML inference (TensorFlow + PyTorch)",
        routes=app.routes,
    )

    openapi_schema["info"]["contact"] = {
        "name": "ML Team",
        "email": "ml-team@example.com"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ==========================================
# EXCEPTION HANDLERS
# ==========================================

@app.exception_handler(ModelLoadingError)
async def model_loading_error_handler(
    request: Request,
    exc: ModelLoadingError
):
    return JSONResponse(
        status_code=500,
        content={
            "error": "ModelLoadingError",
            "message": exc.message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request,
    exc: PredictionError
):
    return JSONResponse(
        status_code=500,
        content={
            "error": "PredictionError",
            "message": exc.message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# GLOBAL OBJECTS
# ==========================================

loader = ModelLoader()
prediction_store: Dict[str, dict] = {}


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get(
    "/health",
    summary="Health Check",
    description="Returns API health status and service availability"
)
async def health():
    return {"status": "healthy"}


# ==========================================
# BACKGROUND PREDICTION TASK
# ==========================================

async def process_prediction(request_id: str, data: dict):
    try:
        await asyncio.sleep(2)

        features = data["features"]
        model_type = data["model_type"]

        result = sum(features) / len(features)

        prediction_store[request_id]["status"] = "completed"
        prediction_store[request_id]["result"] = {
            "prediction": result,
            "model_type": model_type,
            "status": "success"
        }

    except Exception as e:
        raise PredictionError(
            f"Prediction processing failed: {str(e)}"
        )


# ==========================================
# ASYNC PREDICTION ENDPOINT
# ==========================================

@app.post(
    "/predict-async",
    status_code=202,
    summary="Async Prediction",
    description="Submit a prediction request. Returns request_id for later retrieval."
)
async def async_predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    try:
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

    except Exception as e:
        raise PredictionError(
            f"Failed to start prediction: {str(e)}"
        )


# ==========================================
# GET PREDICTION RESULT
# ==========================================

@app.get(
    "/predict-result/{request_id}",
    summary="Get Prediction Result",
    description="Retrieve prediction result using request_id"
)
async def get_result(request_id: str):
    try:

        if request_id not in prediction_store:
            raise PredictionError(
                "Request ID not found"
            )

        return prediction_store[request_id]

    except PredictionError:
        raise

    except Exception as e:
        raise PredictionError(
            f"Failed to retrieve prediction result: {str(e)}"
        )