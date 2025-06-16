from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str
    version: str

class PredictResponse(BaseModel):
    positive_probability: float
    negative_probability: float
    sentiment: str
