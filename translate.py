import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NMTTranslator:
    def __init__(self, model_path: str = "./results"):
        print(f"Loading model from {model_path}...")
        try:
            # We load the tokenizer and model manually
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            # Move to GPU if available (for your Colab run later)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            print(f"Model loaded successfully on {self.device}!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def translate(self, text: str) -> str:
        """
        Manually handles the tokenization, generation, and decoding.
        This is more robust than using the pipeline() wrapper.
        """
        # 1. Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 2. Generate the translation using the model
        # max_new_tokens is now preferred over max_length in 2026 versions
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )

        # 3. Decode the tokens back into a string
        translated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True)
        return translated_text


if __name__ == "__main__":
    translator = NMTTranslator()
    sample_text = "What don't I know?"
    print(f"Input: {sample_text}")
    print(f"Output: {translator.translate(sample_text)}")
