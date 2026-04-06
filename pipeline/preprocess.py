import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from core.text_utils import clean_text
from config import settings


class DataPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_checkpoint)

    def process_batch(self, examples):
        inputs = [clean_text(ex["en"]) for ex in examples["translation"]]
        targets = [clean_text(ex["fr"]) for ex in examples["translation"]]

        return self.tokenizer(
            inputs,
            text_target=targets,
            max_length=settings.max_length,
            truncation=True,
            padding="max_length"
        )


def run_preprocessing(subset_size: int):
    dataset = load_dataset("opus100", "en-fr", split=f"train[:{subset_size}]")
    preprocessor = DataPreprocessor()

    tokenized_dataset = dataset.map(
        preprocessor.process_batch,
        batched=True,
        remove_columns=dataset.column_names
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    os.makedirs(settings.processed_data_path, exist_ok=True)
    split_dataset.save_to_disk(settings.processed_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100)
    args = parser.parse_args()
    run_preprocessing(args.subset)
