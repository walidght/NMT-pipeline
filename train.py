from dotenv import load_dotenv
from huggingface_hub import login
import os
import numpy as np
import evaluate
import wandb
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

load_dotenv()

login(os.environ["HF_TOKEN"])

# --- Configuration ---
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-fr"
DATA_PATH = "./processed_data"
HF_USERNAME = "walidght"  # <-- CHANGE THIS TO YOUR HF USERNAME
HUB_MODEL_ID = f"{HF_USERNAME}/linguistflow-en-fr-mvp"
WANDB_PROJECT = "linguistflow-nmt"


def main():
    # 1. Initialize W&B
    wandb.init(project=WANDB_PROJECT, name="marian-mt-finetune")

    # 2. Load Processed Data & Tokenizer
    print("Loading data from disk...")
    dataset = load_from_disk(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # 3. Setup Metric (SacreBLEU)
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        """Decodes predictions and labels to compute the BLEU score."""
        preds, labels = eval_preds

        # Replace -100 in labels (used for padding) with pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and references
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # SacreBLEU expects references as a list of lists
        decoded_labels = [[label] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels)
        return {"bleu": result["score"]}

    # 4. Define Training Arguments
    # These parameters are optimized for a weekend run (fast, minimal epochs)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=200,             # Evaluate every 200 steps
        logging_steps=50,           # Log loss every 50 steps
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,         # Only keep the last 2 checkpoints
        num_train_epochs=3,         # Keep it low for the MVP
        predict_with_generate=True,  # Required for BLEU calculation
        report_to="wandb",          # Logs metrics automatically to W&B
        push_to_hub=True,           # Pushes to HF Hub at the end
        hub_model_id=HUB_MODEL_ID
    )

    # The Data Collator handles dynamic padding for batches during training
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6. Train the Model
    print("Starting training...")
    trainer.train()

    # 7. Generate W&B Validation Table
    print("Generating Validation Table for W&B...")
    generate_wandb_table(trainer, tokenizer, dataset["test"])

    # 8. Push to Hugging Face Hub
    print("Pushing model to Hugging Face Hub...")
    trainer.push_to_hub()

    # Close W&B run
    wandb.finish()
    print("✅ Training complete and model pushed to Hub!")


def generate_wandb_table(trainer, tokenizer, eval_dataset, num_samples=50):
    """
    Runs inference and logs a visual table to W&B.
    Includes type-safety checks to prevent 'int() argument must be string/list' errors.
    """
    # 1. Defensive sampling (as we did before)
    actual_num_samples = min(num_samples, len(eval_dataset))
    sample_dataset = eval_dataset.shuffle(
        seed=42).select(range(actual_num_samples))

    # 2. Generate predictions
    print(f"Generating predictions for {actual_num_samples} samples...")
    # trainer.predict returns a NamedTuple; we want the .predictions (numpy array)
    output = trainer.predict(sample_dataset, metric_key_prefix="predict")
    predictions = output.predictions

    # Seq2Seq models sometimes return a tuple (preds, extras); we only want preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # 3. TYPE SAFETY: Convert everything to plain Python lists
    # This prevents the "TypeError: int() argument... not 'list'" error
    try:
        # Extract inputs and labels, converting to standard lists
        raw_inputs = [list(map(int, ids))
                      for ids in sample_dataset["input_ids"]]

        # Labels need -100 replaced by pad_token_id before decoding
        raw_labels = []
        for ids in sample_dataset["labels"]:
            cleaned_ids = [
                int(i) if i != -100 else tokenizer.pad_token_id for i in ids]
            raw_labels.append(cleaned_ids)

        # Predictions (already a numpy array) to list
        raw_preds = predictions.tolist()

        # 4. Decode all at once
        inputs_decoded = tokenizer.batch_decode(
            raw_inputs, skip_special_tokens=True)
        actuals_decoded = tokenizer.batch_decode(
            raw_labels, skip_special_tokens=True)
        preds_decoded = tokenizer.batch_decode(
            raw_preds, skip_special_tokens=True)

        # 5. Log to W&B
        table = wandb.Table(
            columns=["English (Input)", "French (Predicted)", "French (Actual)"])
        for inp, pred, act in zip(inputs_decoded, preds_decoded, actuals_decoded):
            table.add_data(inp, pred, act)

        wandb.log({"Validation Translation Samples": table})
        print("✅ Validation Table successfully logged to W&B!")

    except Exception as e:
        print(
            f"⚠️ Warning: Could not log W&B table due to decoding error: {e}")


if __name__ == "__main__":
    main()
