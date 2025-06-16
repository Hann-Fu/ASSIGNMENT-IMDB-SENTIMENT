# /predict
from fastapi import APIRouter, HTTPException
from schema.prediction_schema import PredictRequest, PredictResponse
from services.inference import sentiment_inference_baseline, sentiment_inference_bert
from utils.logging_decorators import log_request_response_time
from services.inference import sentiment_inference_llm
from exceptions.llm import LLMRateLimitError, LLMServiceError

router = APIRouter()

@router.post("/predict/baseline", response_model=PredictResponse)
@log_request_response_time
def predict_baseline(request: PredictRequest):
    """
    Predict the sentiment of the text, return the sentiment and the probability of the sentiment.

    Args:
        request (PredictRequest): The request object containing the input text.

    Returns:
        PredictResponse: The response object containing the sentiment and probability.
    """
    try:
        if request.version == "1.0.0":
            probability = sentiment_inference_baseline(request.text, request.version)
            response = PredictResponse(
                positive_probability=probability,
                negative_probability=1-probability,
                sentiment="positive" if probability > 0.5 else "negative"
            )
            return response
        else:
            raise HTTPException(status_code=400, detail="Invalid version.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model files are missing or not deployed: {e}")
    except MemoryError as e:
        raise HTTPException(status_code=507, detail=f"Insufficient memory to process the request: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Prediction service temporarily unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/predict/bert", response_model=PredictResponse)
@log_request_response_time
def predict_bert(request: PredictRequest):
    """
    Predict the sentiment of the text using DistilBERT model, return the sentiment and the probability of the sentiment.

    Args:
        request (PredictRequest): The request object containing the input text.

    Returns:
        PredictResponse: The response object containing the sentiment and probability.
    """
    try:
        if request.version == "1.0.0":
            probability = sentiment_inference_bert(request.text, request.version)
            response = PredictResponse(
                positive_probability=probability,
                negative_probability=1-probability,
                sentiment="positive" if probability > 0.5 else "negative"
            )
            return response
        else:
            raise HTTPException(status_code=400, detail="Invalid version.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model files are missing or not deployed: {e}")
    except MemoryError as e:
        raise HTTPException(status_code=507, detail=f"Insufficient memory to process the request: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Prediction service temporarily unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/predict/llm", response_model=PredictResponse)
@log_request_response_time
async def predict_llm(request: PredictRequest):
    """
    Use LLM to predict the sentiment of the text, return the sentiment and the probability of the sentiment.

    Args:
        request (PredictRequest): The request object containing the input text.

    Returns:
        PredictResponse: The response object containing the sentiment and probability.
    """
    try:
        if request.version == "1.0.0":
            result = await sentiment_inference_llm(request.text, request.version)
            response = PredictResponse(
                positive_probability=1 if result == "positive" else 0,
                negative_probability=1 if result == "negative" else 0,
                sentiment=result
            )
            return response
        else:
            raise HTTPException(status_code=400, detail="Invalid version.")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except LLMServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")



