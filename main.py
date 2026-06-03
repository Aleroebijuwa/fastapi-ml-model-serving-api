from fastapi import FastAPI
import uvicorn
import torch
import tensorflow as tf
from pydantic import BaseModel
git 
app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready API for serving ML models",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "torch_version": torch.__version__,
        "tensorflow_version": tf.__version__
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)