from pydantic import BaseModel, Field, validator
from typing import Literal, List


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of exactly 10 numerical features")

    model_type: Literal["tensorflow", "pytorch"] = Field(
        default="tensorflow",
        description="Type of model to use for prediction"
    )

    
    @validator("features")
    def validate_features_length(cls, v):
        if len(v) != 10:
            raise ValueError(f"Exactly 10 features required, got {len(v)}")
        return v

    
    @validator("features", each_item=True)
    def validate_feature_values(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("All features must be numeric")
        return float(v)  # sanitization: force float


class PredictionResponse(BaseModel):
    prediction: float
    model_type: str
    status: str