"""
Arabic OCR Training Pipeline using DeepSeek-OCR and Unsloth.

This pipeline trains Arabic OCR models using the DeepSeek-OCR architecture
with LoRA fine-tuning, adapted from the working notebook implementation.
Uses Metaflow for orchestration and Weights & Biases for experiment tracking.
"""

import os
import tempfile
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from metaflow import FlowSpec, Parameter, card, current, environment, project, step
from transformers import Trainer, TrainingArguments
from unsloth import is_bf16_supported

from .data_collator import DeepSeekOCRDataCollator, convert_to_conversation
from .metrics import calculate_cer, calculate_wer, evaluate_batch
from .model import (
    load_deepseek_ocr_model,
    prepare_model_for_training,
    save_model_and_tokenizer,
    setup_lora_model,
)


@project(name="arabic-ocr")
class ArabicOCRTraining(FlowSpec):
    """
    Arabic OCR training pipeline using DeepSeek-OCR with LoRA fine-tuning.

    This pipeline loads the Arabic OCR dataset, fine-tunes DeepSeek-OCR model,
    evaluates performance, and saves the trained model.
    """

    # Weights & Biases configuration
    wandb_project = Parameter(
        "wandb-project",
        help="Weights & Biases project name for experiment tracking.",
        default="arabic-ocr-training",
    )

    wandb_run_name = Parameter(
        "wandb-run-name",
        help="Weights & Biases run name (optional).",
        default=None,
    )

    # Dataset parameters
    dataset_name = Parameter(
        "dataset-name",
        help="HuggingFace dataset name for Arabic OCR.",
        default="mssqpi/Arabic-OCR-Dataset",
    )

    dataset_size = Parameter(
        "dataset-size",
        help="Number of samples to use for training (0 = all).",
        default=1000,
    )

    train_test_split = Parameter(
        "train-test-split",
        help="Percentage of data to use for training (rest for evaluation).",
        default=0.8,
    )

    # Model parameters
    model_path = Parameter(
        "model-path",
        help="Path to DeepSeek-OCR model (will download if not found).",
        default="./deepseek_ocr",
    )

    load_in_4bit = Parameter(
        "load-in-4bit",
        help="Load model in 4-bit quantization to save memory.",
        default=False,
    )

    # LoRA parameters
    lora_r = Parameter(
        "lora-r",
        help="LoRA rank (higher = more capacity, risk of overfitting).",
        default=16,
    )

    lora_alpha = Parameter(
        "lora-alpha",
        help="LoRA alpha parameter.",
        default=16,
    )

    lora_dropout = Parameter(
        "lora-dropout",
        help="LoRA dropout rate.",
        default=0.0,
    )

    # Training parameters
    per_device_train_batch_size = Parameter(
        "batch-size",
        help="Training batch size per device.",
        default=2,
    )

    gradient_accumulation_steps = Parameter(
        "gradient-accumulation",
        help="Number of steps to accumulate gradients.",
        default=4,
    )

    learning_rate = Parameter(
        "learning-rate",
        help="Learning rate for training.",
        default=2e-4,
    )

    num_train_epochs = Parameter(
        "epochs",
        help="Number of training epochs.",
        default=3,
    )

    max_steps = Parameter(
        "max-steps",
        help="Maximum training steps (overrides epochs if set).",
        default=-1,
    )

    warmup_steps = Parameter(
        "warmup-steps",
        help="Number of warmup steps.",
        default=10,
    )

    # Evaluation parameters
    eval_steps = Parameter(
        "eval-steps",
        help="Number of steps between evaluations.",
        default=100,
    )

    # Output parameters
    output_dir = Parameter(
        "output-dir",
        help="Directory to save trained model and outputs.",
        default="./outputs/arabic_ocr",
    )

    push_to_hub = Parameter(
        "push-to-hub",
        help="Push trained model to HuggingFace Hub.",
        default=False,
    )

    hub_model_name = Parameter(
        "hub-model-name",
        help="HuggingFace Hub model name for uploading.",
        default=None,
    )

    hub_token = Parameter(
        "hub-token",
        help="HuggingFace Hub token for uploading.",
        default=None,
    )

    @step
    def start(self):
        """Initialize the Arabic OCR training pipeline."""
        print("🕌 Starting Arabic OCR Training Pipeline")
        print("=" * 50)

        # Initialize Weights & Biases
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            config={
                "dataset_name": self.dataset_name,
                "dataset_size": self.dataset_size,
                "model_path": self.model_path,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "learning_rate": self.learning_rate,
                "batch_size": self.per_device_train_batch_size,
                "epochs": self.num_train_epochs,
                "max_steps": self.max_steps,
            },
        )

        print(f"📊 W&B Project: {self.wandb_project}")
        print(f"📖 Dataset: {self.dataset_name}")
        print(f"📈 Training samples: {self.dataset_size}")

        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        """Load and prepare the Arabic OCR dataset."""
        print("📚 Loading Arabic OCR Dataset...")

        # Load dataset
        if self.dataset_size > 0:
            dataset = load_dataset(
                self.dataset_name, split=f"train[:{self.dataset_size}]"
            )
        else:
            dataset = load_dataset(self.dataset_name, split="train")

        print(f"✅ Loaded {len(dataset)} samples")

        # Convert to conversation format
        print("🔄 Converting to conversation format...")
        instruction = "<image>\nFree OCR. "
        converted_dataset = [convert_to_conversation(sample, instruction) for sample in dataset]

        # Split into train/eval
        split_idx = int(len(converted_dataset) * self.train_test_split)
        self.train_dataset = converted_dataset[:split_idx]
        self.eval_dataset = converted_dataset[split_idx:]

        print(f"📊 Train samples: {len(self.train_dataset)}")
        print(f"📊 Eval samples: {len(self.eval_dataset)}")

        # Log sample to W&B
        wandb.log({
            "dataset_size": len(dataset),
            "train_size": len(self.train_dataset),
            "eval_size": len(self.eval_dataset),
        })

        self.next(self.setup_model)

    @step
    def setup_model(self):
        """Load and configure the DeepSeek-OCR model with LoRA."""
        print("🤖 Setting up DeepSeek-OCR Model...")

        # Load base model
        self.model, self.tokenizer = load_deepseek_ocr_model(
            model_path=self.model_path,
            load_in_4bit=self.load_in_4bit,
            download_if_missing=True,
        )

        # Setup LoRA
        self.model = setup_lora_model(
            self.model,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

        # Prepare for training
        self.model = prepare_model_for_training(self.model)

        # Setup data collator
        self.data_collator = DeepSeekOCRDataCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            image_size=640,
            base_size=1024,
            crop_mode=True,
            train_on_responses_only=True,
        )

        print("✅ Model setup complete!")

        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the Arabic OCR model using the configured parameters."""
        print("🚀 Starting Model Training...")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.num_train_epochs if self.max_steps == -1 else None,
            max_steps=self.max_steps if self.max_steps > 0 else None,
            learning_rate=self.learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps" if len(self.eval_dataset) > 0 else "no",
            eval_steps=self.eval_steps if len(self.eval_dataset) > 0 else None,
            save_strategy="steps",
            save_steps=self.eval_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to="wandb",  # Use W&B for logging
            run_name=self.wandb_run_name,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if len(self.eval_dataset) > 0 else None,
            args=training_args,
        )

        # Start training
        print("🏃 Training started...")
        training_stats = trainer.train()

        # Save training statistics
        self.training_time = training_stats.metrics["train_runtime"]
        self.final_loss = training_stats.metrics["train_loss"]

        print(f"✅ Training completed in {self.training_time:.2f} seconds")
        print(f"📉 Final loss: {self.final_loss:.4f}")

        # Log to W&B
        wandb.log({
            "training_time": self.training_time,
            "final_loss": self.final_loss,
        })

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """Evaluate the trained model on the test set."""
        print("📊 Evaluating Model Performance...")

        if len(self.eval_dataset) == 0:
            print("⚠️ No evaluation dataset available. Skipping evaluation.")
            self.eval_results = {}
            self.next(self.save_model)
            return

        # Prepare model for inference
        from .model import prepare_model_for_inference
        eval_model = prepare_model_for_inference(self.model)

        # Run evaluation on a subset of eval data for speed
        eval_sample_size = min(50, len(self.eval_dataset))
        eval_samples = self.eval_dataset[:eval_sample_size]

        predictions = []
        ground_truths = []

        print(f"🔍 Evaluating on {eval_sample_size} samples...")

        for sample in eval_samples:
            try:
                # Get image and ground truth text
                image = sample["messages"][0]["images"][0]
                ground_truth = sample["messages"][1]["content"]

                # Save image temporarily for inference
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    temp_image_path = tmp_file.name

                # Run inference
                prompt = "<image>\nFree OCR. "
                result = eval_model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path="temp_output",
                    image_size=640,
                    base_size=1024,
                    crop_mode=True,
                    save_results=False,
                )

                predictions.append(result.strip())
                ground_truths.append(ground_truth.strip())

                # Clean up temp file
                os.unlink(temp_image_path)

            except Exception as e:
                print(f"❌ Error during inference: {e}")
                continue

        # Calculate metrics
        if predictions and ground_truths:
            self.eval_results = evaluate_batch(predictions, ground_truths)

            print("📈 Evaluation Results:")
            print(f"  Character Error Rate (CER): {self.eval_results['cer']:.3f}")
            print(f"  Word Error Rate (WER):      {self.eval_results['wer']:.3f}")
            print(f"  BLEU Score:                 {self.eval_results['bleu']:.3f}")
            print(f"  Exact Match:                {self.eval_results['exact_match']:.3f}")

            # Log to W&B
            wandb.log({
                "eval_cer": self.eval_results["cer"],
                "eval_wer": self.eval_results["wer"],
                "eval_bleu": self.eval_results["bleu"],
                "eval_exact_match": self.eval_results["exact_match"],
            })

            # Log examples to W&B
            examples = []
            for i, (pred, gt) in enumerate(zip(predictions[:5], ground_truths[:5])):
                examples.append({
                    "sample": i,
                    "prediction": pred,
                    "ground_truth": gt,
                    "cer": calculate_cer(pred, gt),
                    "wer": calculate_wer(pred, gt),
                })
            wandb.log({"evaluation_examples": wandb.Table(data=examples)})
        else:
            print("❌ No successful evaluations completed")
            self.eval_results = {}

        self.next(self.save_model)

    @step
    def save_model(self):
        """Save the trained model and upload to HuggingFace Hub if requested."""
        print("💾 Saving Trained Model...")

        # Create model save directory
        model_save_path = Path(self.output_dir) / "trained_model"
        model_save_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        save_model_and_tokenizer(
            model=self.model,
            tokenizer=self.tokenizer,
            save_path=str(model_save_path),
            push_to_hub=self.push_to_hub,
            hub_model_name=self.hub_model_name,
            token=self.hub_token,
        )

        print(f"✅ Model saved to {model_save_path}")

        if self.push_to_hub and self.hub_model_name:
            print(f"🌐 Model uploaded to HuggingFace Hub: {self.hub_model_name}")
            wandb.log({"hub_model_name": self.hub_model_name})

        self.next(self.end)

    @step
    def end(self):
        """Complete the Arabic OCR training pipeline."""
        print("🎉 Arabic OCR Training Pipeline Complete!")
        print("=" * 50)

        # Summary
        print("📋 Training Summary:")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Training samples: {len(self.train_dataset)}")
        if hasattr(self, "training_time"):
            print(f"  Training time: {self.training_time:.2f} seconds")
            print(f"  Final loss: {self.final_loss:.4f}")

        if hasattr(self, "eval_results") and self.eval_results:
            print(f"  Character Error Rate: {self.eval_results['cer']:.3f}")
            print(f"  Word Error Rate: {self.eval_results['wer']:.3f}")
            print(f"  BLEU Score: {self.eval_results['bleu']:.3f}")

        print(f"  Model saved: {self.output_dir}")

        # Log final summary to W&B
        summary = {
            "pipeline_completed": True,
            "final_model_path": self.output_dir,
        }

        if hasattr(self, "training_time"):
            summary.update({
                "total_training_time": self.training_time,
                "final_training_loss": self.final_loss,
            })

        if hasattr(self, "eval_results"):
            summary.update(self.eval_results)

        wandb.log(summary)
        wandb.finish()

        print("🕌 Arabic OCR training pipeline completed successfully!")


if __name__ == "__main__":
    ArabicOCRTraining()