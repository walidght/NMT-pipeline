import gradio as gr
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, constr
from services.translator import TranslatorService
from config import settings

translator_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global translator_instance
    try:
        translator_instance = TranslatorService(settings.local_model_path)
    except Exception as e:
        print(f"Startup Warning: Model failed to load. {e}")
    yield
    translator_instance = None

app = FastAPI(title="LinguistFlow API", lifespan=lifespan)

# Dependencies


def get_translator() -> TranslatorService:
    if not translator_instance:
        raise HTTPException(
            status_code=503, detail="Translation service unavailable.")
    return translator_instance

# Schemas


class TranslationRequest(BaseModel):
    # Validates empty strings instantly
    text: constr(strip_whitespace=True, min_length=1)


class TranslationResponse(BaseModel):
    input_text: str
    translated_text: str

# Endpoints


@app.get("/health")
def health_check(translator: TranslatorService = Depends(get_translator)):
    return {"status": "healthy"}


@app.post("/translate", response_model=TranslationResponse)
def translate_text(
    request: TranslationRequest,
    translator: TranslatorService = Depends(get_translator)
):
    try:
        translated = translator.translate(request.text)
        return TranslationResponse(input_text=request.text, translated_text=translated)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Internal translation error.")


description = """
### 🌍 LinguistFlow NMT
Translate English to French using a fine-tuned MarianMT transformer. 
*API endpoints are available at `/docs`.*
"""

demo = gr.Interface(
    fn=gradio_translate,
    inputs=gr.Textbox(label="English Input", lines=3,
                      placeholder="Type something..."),
    outputs=gr.Textbox(label="French Translation"),
    title="LinguistFlow",
    description=description,
    examples=[["Hello, how are you today?"], [
        "The model is running in the cloud."]],
    allow_flagging="never"
)

# Mount Gradio to the FastAPI app at the root route
app = gr.mount_gradio_app(app, demo, path="/")
