import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_baseline_endpoint_invalid_version():
    """Test baseline endpoint with invalid version"""
    response = client.post(
        "/api/v1/predict/baseline",
        json={"text": "This is a test", "version": "2.0.0"}
    )
    assert response.status_code == 500


def test_baseline_endpoint_missing_text():
    """Test baseline endpoint with missing text"""
    response = client.post(
        "/api/v1/predict/baseline",
        json={"version": "1.0.0"}
    )
    assert response.status_code == 422


def test_bert_endpoint_invalid_version():
    """Test BERT endpoint with invalid version"""
    response = client.post(
        "/api/v1/predict/bert",
        json={"text": "This is a test", "version": "2.0.0"}
    )
    assert response.status_code == 500


def test_llm_endpoint_invalid_version():
    """Test LLM endpoint with invalid version"""
    response = client.post(
        "/api/v1/predict/llm",
        json={"text": "This is a test", "version": "2.0.0"}
    )
    assert response.status_code == 500


def test_baseline_endpoint_empty_text():
    """Test baseline endpoint with empty text"""
    response = client.post(
        "/api/v1/predict/baseline",
        json={"text": "", "version": "1.0.0"}
    )
    assert response.status_code == 400 