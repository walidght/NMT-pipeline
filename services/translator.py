import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import settings


class TranslatorService:
    def __init__(self, model_path: str = settings.local_model_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path).to(self.device)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            num_beams=settings.beam_size,
            early_stopping=True
        )

        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
