import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_ask_without_document():
    # If we ask a question about something out of context, it should fallback properly.
    response = client.post("/ask", json={"query": "How do I bake a chocolate cake?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert data["confidence"] == "low"
    assert len(data["sources"]) == 0
    assert "not available" in data["answer"].lower() or "not" in data["answer"].lower()

def test_upload_endpoint_no_file():
    # Test uploading without a file
    response = client.post("/upload")
    assert response.status_code == 422 # Unprocessable Entity
