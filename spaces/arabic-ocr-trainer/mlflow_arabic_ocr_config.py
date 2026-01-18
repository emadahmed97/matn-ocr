#!/usr/bin/env python3
"""
MLflow configuration and logging utilities for Arabic OCR experiments.

This module provides utilities to configure MLflow for OCR experiments,
log OCR-specific metrics, and track model performance.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Import Arabic OCR metrics directly to avoid dependencies
import sys
sys.path.append('pipelines/arabic_ocr')

# Import metrics without the model dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("metrics", "pipelines/arabic_ocr/metrics.py")
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)

calculate_cer = metrics_module.calculate_cer
calculate_wer = metrics_module.calculate_wer
calculate_bleu = metrics_module.calculate_bleu
evaluate_batch = metrics_module.evaluate_batch
format_evaluation_results = metrics_module.format_evaluation_results


class ArabicOCRExperiment:
    """MLflow experiment manager for Arabic OCR tasks."""

    def __init__(self,
                 experiment_name: str = "arabic-ocr-nougat",
                 tracking_uri: str = None):
        """
        Initialize Arabic OCR experiment tracking.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        self.client = MlflowClient(self.tracking_uri)
        self.current_run = None

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """
        Start a new MLflow run for OCR experiment.

        Args:
            run_name: Name for the run
            tags: Additional tags for the run

        Returns:
            Run ID
        """
        default_tags = {
            "model_type": "nougat-small",
            "task": "arabic-ocr",
            "language": "arabic",
            "domain": "classical-islamic-texts"
        }

        if tags:
            default_tags.update(tags)

        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags
        )

        return self.current_run.info.run_id

    def log_dataset_info(self,
                        dataset_name: str,
                        num_samples: int,
                        num_books: int = None,
                        avg_text_length: float = None,
                        diacritic_ratio: float = None):
        """
        Log dataset information.

        Args:
            dataset_name: Name of the dataset
            num_samples: Number of samples in dataset
            num_books: Number of books (if applicable)
            avg_text_length: Average text length
            diacritic_ratio: Ratio of diacritics to text
        """
        params = {
            "dataset_name": dataset_name,
            "num_samples": num_samples,
        }

        metrics = {}

        if num_books is not None:
            params["num_books"] = num_books

        if avg_text_length is not None:
            metrics["avg_text_length"] = avg_text_length

        if diacritic_ratio is not None:
            metrics["diacritic_ratio"] = diacritic_ratio

        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

    def log_model_config(self,
                        model_name: str = "nougat-small",
                        image_size: tuple = (896, 672),
                        max_length: int = 4096,
                        learning_rate: float = 1e-4,
                        batch_size: int = 1,
                        **kwargs):
        """
        Log model configuration parameters.

        Args:
            model_name: Name of the base model
            image_size: Input image dimensions
            max_length: Maximum text sequence length
            learning_rate: Training learning rate
            batch_size: Training batch size
            **kwargs: Additional configuration parameters
        """
        params = {
            "model_name": model_name,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "max_length": max_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }

        # Add any additional parameters
        params.update(kwargs)

        mlflow.log_params(params)

    def log_training_metrics(self,
                           epoch: int,
                           train_loss: float,
                           val_loss: float = None,
                           learning_rate: float = None):
        """
        Log training metrics for current epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Current learning rate
        """
        metrics = {"train_loss": train_loss}

        if val_loss is not None:
            metrics["val_loss"] = val_loss

        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        mlflow.log_metrics(metrics, step=epoch)

    def log_ocr_evaluation(self,
                          predictions: List[str],
                          ground_truths: List[str],
                          step: Optional[int] = None,
                          prefix: str = ""):
        """
        Log OCR evaluation metrics.

        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts
            step: Step number for time series logging
            prefix: Prefix for metric names (e.g., "val_", "test_")
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        # Calculate metrics using our Arabic OCR utilities
        results = evaluate_batch(predictions, ground_truths)

        # Prepare metrics with prefix
        metrics = {}
        for key, value in results.items():
            metric_name = f"{prefix}{key}" if prefix else key
            metrics[metric_name] = value

        # Log to MLflow
        if step is not None:
            mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics)

        return results

    def log_sample_predictions(self,
                             predictions: List[str],
                             ground_truths: List[str],
                             images: List[Any] = None,
                             max_samples: int = 5):
        """
        Log sample predictions for qualitative analysis.

        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts
            images: List of input images (optional)
            max_samples: Maximum number of samples to log
        """
        import arabic_reshaper
        from bidi.algorithm import get_display

        num_samples = min(len(predictions), len(ground_truths), max_samples)

        samples = []
        for i in range(num_samples):
            pred = predictions[i]
            gt = ground_truths[i]

            # Format Arabic text for display
            try:
                formatted_pred = get_display(arabic_reshaper.reshape(pred)) if pred else ""
                formatted_gt = get_display(arabic_reshaper.reshape(gt)) if gt else ""
            except:
                formatted_pred = pred
                formatted_gt = gt

            # Calculate individual metrics
            cer = calculate_cer(pred, gt)
            wer = calculate_wer(pred, gt)
            bleu = calculate_bleu(pred, gt)

            sample = {
                "sample_id": i,
                "predicted_text": formatted_pred,
                "ground_truth_text": formatted_gt,
                "cer": cer,
                "wer": wer,
                "bleu": bleu
            }

            samples.append(sample)

        # Log as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("# Arabic OCR Sample Predictions\n\n")
            for sample in samples:
                f.write(f"## Sample {sample['sample_id']}\n")
                f.write(f"**Ground Truth**: {sample['ground_truth_text']}\n")
                f.write(f"**Predicted**:   {sample['predicted_text']}\n")
                f.write(f"**CER**: {sample['cer']:.3f} | **WER**: {sample['wer']:.3f} | **BLEU**: {sample['bleu']:.3f}\n\n")

            temp_path = f.name

        mlflow.log_artifact(temp_path, "sample_predictions")
        os.unlink(temp_path)

    def log_model_artifacts(self,
                          model_path: str,
                          tokenizer_path: str = None,
                          processor_path: str = None):
        """
        Log model artifacts to MLflow.

        Args:
            model_path: Path to saved model
            tokenizer_path: Path to tokenizer
            processor_path: Path to image processor
        """
        mlflow.log_artifact(model_path, "model")

        if tokenizer_path:
            mlflow.log_artifact(tokenizer_path, "tokenizer")

        if processor_path:
            mlflow.log_artifact(processor_path, "processor")

    def finish_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            self.current_run = None

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple runs and return metrics comparison.

        Args:
            run_ids: List of MLflow run IDs to compare

        Returns:
            Dictionary with comparison data
        """
        comparison = {}

        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                "name": run.data.tags.get("mlflow.runName", run_id[:8]),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "status": run.info.status
            }

        return comparison


def test_arabic_ocr_mlflow():
    """Test Arabic OCR MLflow integration with sample data."""

    print("🔬 Testing Arabic OCR MLflow Integration")
    print("=" * 50)

    # Initialize experiment
    experiment = ArabicOCRExperiment()

    # Start a test run
    run_id = experiment.start_run(
        run_name="test-arabic-ocr-integration",
        tags={"test": "true", "purpose": "validation"}
    )

    print(f"✅ Started MLflow run: {run_id}")

    try:
        # Log dataset info
        experiment.log_dataset_info(
            dataset_name="MohamedRashad/arabic-books",
            num_samples=1000,
            num_books=8647,
            avg_text_length=250.5,
            diacritic_ratio=0.12
        )
        print("✅ Logged dataset information")

        # Log model config
        experiment.log_model_config(
            model_name="microsoft/nougat-small",
            image_size=(896, 672),
            max_length=4096,
            learning_rate=5e-5,
            batch_size=2
        )
        print("✅ Logged model configuration")

        # Simulate training metrics
        for epoch in range(3):
            experiment.log_training_metrics(
                epoch=epoch,
                train_loss=2.5 - (epoch * 0.3),
                val_loss=2.8 - (epoch * 0.25),
                learning_rate=5e-5 * (0.95 ** epoch)
            )
        print("✅ Logged training metrics")

        # Test OCR evaluation with sample Arabic text
        predictions = [
            "بسم الله الرحمن الرحيم",
            "الحمد لله رب العالمين",
            "كتاب الطهارة"
        ]

        ground_truths = [
            "بسم الله الرحمن الرحيم",
            "الحمد لله رب العالمين",
            "كتاب الطهاره"  # Slight difference to test metrics
        ]

        results = experiment.log_ocr_evaluation(
            predictions, ground_truths,
            step=100, prefix="test_"
        )
        print("✅ Logged OCR evaluation metrics")
        print(format_evaluation_results(results))

        # Log sample predictions
        experiment.log_sample_predictions(predictions, ground_truths)
        print("✅ Logged sample predictions")

        print(f"\n🎯 MLflow Experiment URL: {experiment.tracking_uri}/#/experiments/{experiment.experiment_id}")
        print(f"📊 Run Details: {experiment.tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}")

    finally:
        # Always end the run
        experiment.finish_run()
        print("✅ Finished MLflow run")

    return experiment, run_id


if __name__ == "__main__":
    test_arabic_ocr_mlflow()