import pytest
from pydantic import ValidationError
from schema.prediction_schema import PredictRequest, PredictResponse


def test_predict_request_valid():
    """Test valid request creation"""
    request = PredictRequest(text="Hello", version="1.0.0")
    assert request.text == "Hello"
    assert request.version == "1.0.0"


def test_predict_request_missing_text():
    """Test request fails without text"""
    with pytest.raises(ValidationError):
        PredictRequest(version="1.0.0")


def test_predict_response_valid():
    """Test valid response creation"""
    response = PredictResponse(
        positive_probability=0.8,
        negative_probability=0.2,
        sentiment="positive"
    )
    assert response.positive_probability == 0.8
    assert response.sentiment == "positive"


def test_predict_response_missing_sentiment():
    """Test response fails without sentiment"""
    with pytest.raises(ValidationError):
        PredictResponse(positive_probability=0.8, negative_probability=0.2) 