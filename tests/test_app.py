import pytest
from fastapi.testclient import TestClient
from api.main import app, get_translator
from core.text_utils import clean_text

# 1. Pure Unit Tests


def test_clean_text_removes_noise():
    assert clean_text("<p>Hello \u200b world!</p>") == "Hello world!"
    assert clean_text(None) == ""

# 2. API Integration Tests (using Dependency Injection Mocking)


class MockTranslator:
    def translate(self, text: str) -> str:
        return "Ceci est un test."


# Override the real model with our fake class
app.dependency_overrides[get_translator] = lambda: MockTranslator()
client = TestClient(app)


def test_translate_endpoint_success():
    response = client.post("/translate", json={"text": "This is a test."})
    assert response.status_code == 200
    assert response.json()["translated_text"] == "Ceci est un test."


def test_translate_empty_string():
    response = client.post("/translate", json={"text": "   "})
    assert response.status_code == 422  # FastAPI Pydantic validation catches this now


def test_translate_long_string():
    response = client.post("/translate", json={"text": "word " * 500})
    assert response.status_code == 200
