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
from config import settings


class TranslationMetrics:
    """Encapsulates the BLEU score computation logic."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("sacrebleu")

    def compute(self, eval_preds) -> dict:
        preds, labels = eval_preds

        # Replace -100 with pad_token_id to ignore padding in loss calculation
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # SacreBLEU expects references as a list of lists
        decoded_labels = [[label] for label in decoded_labels]
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels)

        return {"bleu": result["score"]}


class CustomWandbLogger:
    """Handles external observability and visual logging."""
    @staticmethod
    def log_validation_table(trainer: Seq2SeqTrainer, tokenizer, eval_dataset, num_samples: int = 50):
        actual_samples = min(num_samples, len(eval_dataset))
        sample_dataset = eval_dataset.shuffle(
            seed=42).select(range(actual_samples))

        output = trainer.predict(sample_dataset, metric_key_prefix="predict")
        predictions = output.predictions[0] if isinstance(
            output.predictions, tuple) else output.predictions

        try:
            raw_inputs = [list(map(int, ids))
                          for ids in sample_dataset["input_ids"]]
            raw_labels = output.label_ids.tolist()
            raw_preds = predictions.tolist()

            inputs_decoded = tokenizer.batch_decode(
                raw_inputs, skip_special_tokens=True)
            actuals_decoded = tokenizer.batch_decode(
                raw_labels, skip_special_tokens=True)
            preds_decoded = tokenizer.batch_decode(
                raw_preds, skip_special_tokens=True)

            table = wandb.Table(
                columns=["English (Input)", "French (Predicted)", "French (Actual)"])
            for inp, pred, act in zip(inputs_decoded, preds_decoded, actuals_decoded):
                table.add_data(inp, pred, act)

            wandb.log({"Validation Translation Samples": table})
        except Exception as e:
            print(
                f"⚠️ Warning: Could not log W&B table due to decoding error: {e}")


class NMTModelTrainer:
    """Orchestrates data loading, configuration, and the training loop."""

    def __init__(self):
        self.dataset = load_from_disk(settings.processed_data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)
        self.metrics_handler = TranslationMetrics(self.tokenizer)

    def _get_training_args(self) -> Seq2SeqTrainingArguments:
        """Isolates hyperparameter configuration."""
        return Seq2SeqTrainingArguments(
            output_dir=settings.local_model_path,
            eval_strategy="epoch",
            # eval_steps=200,
            logging_steps=50,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=3,
            predict_with_generate=True,
            report_to="wandb",
            push_to_hub=True,
            hub_model_id=settings.hub_model_id
        )

    def execute(self):
        wandb.init(project=settings.wandb_project, name="marian-mt-finetune")

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self._get_training_args(),
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.metrics_handler.compute
        )

        print("Starting training sequence...")
        trainer.train()

        print("Generating validation visuals...")
        CustomWandbLogger.log_validation_table(
            trainer, self.tokenizer, self.dataset["test"])

        print("Pushing artifacts to Hub...")
        trainer.push_to_hub()

        wandb.finish()
        print("✅ Pipeline execution complete!")


if __name__ == "__main__":
    pipeline = NMTModelTrainer()
    pipeline.execute()
