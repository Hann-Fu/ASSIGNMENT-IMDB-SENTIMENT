import pytest
from services.inference import sentiment_inference_baseline, sentiment_inference_bert


def test_baseline_inference_empty_text():
    """Test baseline inference with empty text raises error"""
    with pytest.raises(ValueError):
        sentiment_inference_baseline("", "1.0.0")


def test_baseline_inference_invalid_version():
    """Test baseline inference with invalid version raises error"""
    with pytest.raises(ValueError):
        sentiment_inference_baseline("test text", "2.0.0")


def test_bert_inference_empty_text():
    """Test BERT inference with empty text raises error"""
    with pytest.raises(ValueError):
        sentiment_inference_bert("", "1.0.0")


def test_bert_inference_invalid_version():
    """Test BERT inference with invalid version raises error"""
    with pytest.raises(ValueError):
        sentiment_inference_bert("test text", "2.0.0")


def test_baseline_inference_whitespace():
    """Test baseline inference with whitespace only raises error"""
    with pytest.raises(ValueError):
        sentiment_inference_baseline("   ", "1.0.0") 