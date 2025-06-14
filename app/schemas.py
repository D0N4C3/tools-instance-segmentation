from pydantic import BaseModel, Field
from typing import List

class Prediction(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float
    polygon: List[List[int]]

class PredictionsResponse(BaseModel):
    predictions: List[Prediction]
