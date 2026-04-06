from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from translate import NMTTranslator

# 1. Initialize API and Model
app = FastAPI(
    title="LinguistFlow Translation API",
    description="End-to-End NMT system for English to French.",
    version="1.0.0"
)

# Load the model globally when the app starts
# Note: In production, you might point this to your HF Hub repo: "your-username/linguistflow-mvp"
try:
    translator = NMTTranslator(model_path="./results")
except Exception:
    translator = None

# 2. Define Data Schemas


class TranslationRequest(BaseModel):
    text: str


class TranslationResponse(BaseModel):
    input_text: str
    translated_text: str

# 3. Endpoints


@app.get("/health")
def health_check():
    """
    Health check endpoint. Cloud providers (like AWS or HF Spaces) 
    ping this to ensure your container is alive.
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Model failed to load.")
    return {"status": "healthy", "message": "API is running and model is loaded."}


@app.post("/translate", response_model=TranslationResponse)
def translate_text(request: TranslationRequest):
    """
    Takes an English string and returns the French translation.
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    if not request.text.strip():
        raise HTTPException(
            status_code=400, detail="Input text cannot be empty.")

    try:
        translated = translator.translate(request.text)
        return TranslationResponse(
            input_text=request.text,
            translated_text=translated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Port 7860 is the default required port for Hugging Face Spaces Docker deployments
    uvicorn.run(app, host="0.0.0.0", port=7860)
