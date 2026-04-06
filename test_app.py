import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Import your functions and app
from preprocess import clean_text
from main import app

# Initialize the test client
client = TestClient(app)

# ==========================================
# 1. UNIT TESTS (Preprocessing)
# ==========================================


def test_clean_text_removes_noise():
    """Ensure HTML, extra spaces, and zero-width chars are removed."""
    messy_input = "<p>Hello \u200b   world!  </p>"
    expected_output = "Hello world!"

    assert clean_text(messy_input) == expected_output


def test_clean_text_handles_non_strings():
    """Ensure it doesn't crash on nulls or numbers."""
    assert clean_text(None) == ""
    assert clean_text(123) == ""

# ==========================================
# 2. INTEGRATION TESTS (API & Invariants)
# ==========================================

# We use @patch to "mock" the translator so we don't need real weights loaded


@patch("main.translator")
def test_translate_endpoint_success(mock_translator):
    """Test the API contract: checks 200 OK and JSON keys."""
    # Setup the fake model response
    mock_translator.translate.return_value = "Ceci est un test."

    response = client.post(
        "/translate",
        json={"text": "This is a test."}
    )

    assert response.status_code == 200
    data = response.json()
    assert "input_text" in data
    assert "translated_text" in data
    assert data["translated_text"] == "Ceci est un test."


@patch("main.translator")
def test_translate_empty_string(mock_translator):
    """Invariant: Model should reject empty strings before inference."""
    response = client.post(
        "/translate",
        json={"text": "   "}
    )
    # Our FastAPI app is designed to return a 400 Bad Request for this
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


@patch("main.translator")
def test_translate_long_string(mock_translator):
    """Invariant: API handles very long strings gracefully."""
    mock_translator.translate.return_value = "Texte très long."

    long_text = "word " * 500  # 500 words
    response = client.post(
        "/translate",
        json={"text": long_text}
    )

    assert response.status_code == 200
    assert response.json()["translated_text"] == "Texte très long."
