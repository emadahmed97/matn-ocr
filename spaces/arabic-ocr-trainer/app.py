#!/usr/bin/env python3
"""
Arabic OCR Training Space - HuggingFace Spaces Application

Provides both Gradio UI for manual training and REST API for automation.
Runs on L4 GPU for efficient LoRA fine-tuning.
"""

import os
import sys
import json
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import gradio as gr
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Add the main project to path
sys.path.append(".")
sys.path.append("../ml.school")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CONFIG = {
    "num_samples": 1000,
    "max_steps": 60,
    "model_name": "unsloth/DeepSeek-OCR",
    "dataset_name": "mssqpi/Arabic-OCR-Dataset",
    "learning_rate": 2e-4,
    "batch_size": 2,
    "lora_r": 16,
    "deploy_threshold": 0.05
}

class ArabicOCRTrainingSpace:
    """HuggingFace Spaces training interface for Arabic OCR."""

    def __init__(self):
        self.training_active = False
        self.current_run_id = None
        self.setup_mlflow()

    def setup_mlflow(self):
        """Configure MLflow for the space."""
        import mlflow
        # Use HF Spaces persistent storage
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow configured with URI: {mlflow_uri}")

    def gradio_train(self,
                     num_samples: int,
                     max_steps: int,
                     learning_rate: float,
                     deploy_threshold: float,
                     experiment_name: str) -> str:
        """
        Gradio interface for training.

        Returns:
            Training status and progress information
        """
        if self.training_active:
            return "❌ Training already in progress. Please wait for completion."

        try:
            config = {
                "num_samples": int(num_samples),
                "max_steps": int(max_steps),
                "learning_rate": float(learning_rate),
                "deploy_threshold": float(deploy_threshold),
                "experiment_name": experiment_name,
                "model_name": DEFAULT_CONFIG["model_name"],
                "dataset_name": DEFAULT_CONFIG["dataset_name"],
                "trigger_source": "gradio_ui",
                "timestamp": datetime.now().isoformat()
            }

            return self._execute_training(config)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return f"❌ Training failed: {str(e)}"

    def api_train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        API endpoint for automated training.

        Args:
            config: Training configuration dictionary

        Returns:
            Training result dictionary
        """
        if self.training_active:
            return {
                "success": False,
                "message": "Training already in progress",
                "status": "busy"
            }

        try:
            # Merge with defaults
            full_config = {**DEFAULT_CONFIG, **config}
            full_config["trigger_source"] = "api"
            full_config["timestamp"] = datetime.now().isoformat()

            result = self._execute_training(full_config)

            return {
                "success": True,
                "message": result,
                "run_id": self.current_run_id,
                "status": "completed" if "✅" in result else "failed"
            }

        except Exception as e:
            logger.error(f"API training failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "status": "error"
            }

    def _execute_training(self, config: Dict[str, Any]) -> str:
        """Execute the actual training with the given configuration."""
        self.training_active = True
        self.current_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Import training pipeline
            from pipelines.arabic_ocr_training_pipeline import ArabicOCRTrainer, GPU_AVAILABLE

            # Check GPU availability
            if not GPU_AVAILABLE:
                return "❌ Training failed: GPU not available. This space requires L4 GPU hardware."

            logger.info(f"🚀 Starting training run: {self.current_run_id}")
            logger.info(f"📊 Configuration: {json.dumps(config, indent=2)}")

            # Initialize trainer
            trainer = ArabicOCRTrainer(
                model_name=config["model_name"],
                dataset_name=config["dataset_name"],
                experiment_name=config["experiment_name"],
                output_dir=f"outputs/{self.current_run_id}"
            )

            # Update training config
            trainer.training_config.update({
                "max_steps": config["max_steps"],
                "learning_rate": config["learning_rate"],
                "per_device_train_batch_size": config.get("batch_size", 2)
            })

            progress = []
            progress.append("🔧 Setting up model and tokenizer...")
            yield "\n".join(progress)

            # Setup training
            trainer.load_model_and_tokenizer()
            progress.append("✅ Model loaded")
            yield "\n".join(progress)

            trainer.setup_lora_training()
            progress.append("✅ LoRA adapters configured")
            yield "\n".join(progress)

            dataset_stats = trainer.load_dataset(num_samples=config["num_samples"])
            progress.append(f"✅ Dataset loaded: {dataset_stats['total_samples']} samples")
            yield "\n".join(progress)

            trainer.setup_trainer()
            progress.append("✅ Trainer configured")
            yield "\n".join(progress)

            progress.append("🚀 Starting training...")
            yield "\n".join(progress)

            # Train model
            training_stats = trainer.train(run_name=self.current_run_id)

            # Extract final metrics
            final_metrics = training_stats.metrics
            train_loss = final_metrics.get("train_loss", float("inf"))

            progress.append(f"✅ Training completed!")
            progress.append(f"📊 Final train loss: {train_loss:.4f}")
            progress.append(f"⏱️  Training time: {final_metrics.get('train_runtime', 0):.2f}s")

            # Check deployment threshold
            meets_threshold = train_loss < config["deploy_threshold"]
            if meets_threshold:
                # Save model for deployment
                model_save_path = f"models/{self.current_run_id}"
                trainer.save_model(model_save_path)
                progress.append(f"🚀 Model saved and ready for deployment: {model_save_path}")
                progress.append(f"✅ Performance meets threshold ({train_loss:.4f} < {config['deploy_threshold']})")
            else:
                progress.append(f"⚠️  Model does not meet deployment threshold")
                progress.append(f"   Train loss: {train_loss:.4f} >= {config['deploy_threshold']}")
                progress.append(f"   Model saved for analysis but not deployed")

            return "\n".join(progress)

        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            return f"❌ Training failed: {str(e)}\n\nCheck the logs for details."

        finally:
            self.training_active = False

# Initialize the training space
training_space = ArabicOCRTrainingSpace()

def create_gradio_interface():
    """Create the Gradio interface for manual training."""

    with gr.Blocks(title="Arabic OCR Training", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🕌 Arabic OCR Training Space")
        gr.Markdown("Fine-tune DeepSeek-OCR for Arabic manuscripts using LoRA")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## ⚙️ Training Configuration")

                num_samples = gr.Slider(
                    minimum=10, maximum=10000, value=1000, step=10,
                    label="Number of Training Samples",
                    info="More samples = better quality but longer training"
                )

                max_steps = gr.Slider(
                    minimum=10, maximum=500, value=60, step=10,
                    label="Training Steps",
                    info="60 steps ≈ 10 minutes. More steps = better convergence"
                )

                learning_rate = gr.Slider(
                    minimum=1e-5, maximum=1e-3, value=2e-4, step=1e-5,
                    label="Learning Rate",
                    info="Higher = faster learning but less stable"
                )

                deploy_threshold = gr.Slider(
                    minimum=0.01, maximum=0.2, value=0.05, step=0.01,
                    label="Auto-Deploy Threshold (CER)",
                    info="Deploy model if Character Error Rate < threshold"
                )

                experiment_name = gr.Textbox(
                    value="gradio-arabic-ocr",
                    label="Experiment Name",
                    info="MLflow experiment name for tracking"
                )

            with gr.Column():
                gr.Markdown("## 📊 Training Progress")

                output_text = gr.Textbox(
                    lines=20,
                    label="Training Output",
                    info="Real-time training progress and results"
                )

        with gr.Row():
            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            stop_btn = gr.Button("⛔ Stop Training", variant="secondary", size="lg")

        # Training function that yields progress
        def train_with_progress(*args):
            for progress in training_space.gradio_train(*args):
                yield progress

        train_btn.click(
            fn=train_with_progress,
            inputs=[num_samples, max_steps, learning_rate, deploy_threshold, experiment_name],
            outputs=output_text,
            show_progress=True
        )

        gr.Markdown("""
        ## 💡 Tips
        - **Quick test**: 100 samples, 10 steps (~2 minutes)
        - **Development**: 1000 samples, 60 steps (~10 minutes)
        - **Production**: 5000+ samples, 200+ steps (~30+ minutes)
        - **Cost**: L4 GPU ≈ $0.60/hour
        """)

    return interface


# Create FastAPI app for custom API endpoints
app = FastAPI()

# Create the Gradio interface
demo = create_gradio_interface()

# Add custom API endpoint to FastAPI
@app.post("/api/train")
async def train_api(request: Request):
    """REST API endpoint for automated training."""
    try:
        data = await request.json()

        # Extract parameters
        num_samples = data.get("num_samples", DEFAULT_CONFIG["num_samples"])
        max_steps = data.get("max_steps", DEFAULT_CONFIG["max_steps"])
        deploy_threshold = data.get("deploy_threshold", DEFAULT_CONFIG["deploy_threshold"])
        experiment_name = data.get("experiment_name", f"api-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Create config for training
        config = {
            "num_samples": num_samples,
            "max_steps": max_steps,
            "learning_rate": DEFAULT_CONFIG["learning_rate"],
            "experiment_name": experiment_name,
            "deploy_threshold": deploy_threshold
        }

        # Call the training function
        result = training_space.api_train(config)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"API training failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Launch with API enabled
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,
        show_error=True
    )
