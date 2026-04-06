import re
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-fr"
MAX_LENGTH = 50  # Based on the 95th percentile from your exploration script
DATASET_SIZE = 100 # 10000  # Keeping it small for the weekend MVP
OUTPUT_DIR = "./processed_data"


def clean_text(text):
    """
    Basic text cleaning.
    Note: We DO NOT lowercase here because the Helsinki-NLP model was 
    trained on cased data. Lowercasing would actually degrade performance!
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags if any slipped through
    text = re.sub(r'<[^>]+>', '', text)
    # Remove zero-width spaces and weird unicode whitespace
    text = re.sub(r'[\u200b\u200e\u200f\ufeff]', '', text)
    # Condense multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def main():
    print(f"1. Loading a subset of {DATASET_SIZE} rows...")
    raw_dataset = load_dataset(
        "opus100", "en-fr", split=f"train[:{DATASET_SIZE}]")

    print("2. Loading AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        # Extract the English and French lists from the batch
        inputs = [clean_text(ex["en"]) for ex in examples["translation"]]
        targets = [clean_text(ex["fr"]) for ex in examples["translation"]]

        # Tokenize inputs and targets together
        # The tokenizer automatically handles adding the <eos> (end of sentence) tokens
        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"  # Pad to MAX_LENGTH so tensors are uniform
        )

        return model_inputs

    print("3. Applying cleaning and tokenization (Batched)...")
    # map() applies the function to the dataset. batched=True makes it fast.
    # remove_columns throws away the raw text, keeping only the numeric tensors
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Running tokenizer"
    )

    # 4. Train/Test Split
    # We need a validation set to ensure the model isn't overfitting
    print("4. Splitting into Train and Validation sets...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    # 5. Save to disk
    print(f"5. Saving processed tensors to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_dataset.save_to_disk(OUTPUT_DIR)

    print("\n✅ Preprocessing Complete!")
    print(f"Train set size: {len(split_dataset['train'])}")
    print(f"Validation set size: {len(split_dataset['test'])}")


if __name__ == "__main__":
    main()
