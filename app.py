#!/usr/bin/env python3
"""
Arabic OCR Training & Inference Space - HuggingFace Spaces Application

Provides Gradio UI for manual training and inference, plus REST APIs for automation.
Runs on L4 GPU for efficient LoRA fine-tuning and inference.
"""

# Import Unsloth FIRST before any other ML libraries to avoid import order warnings
try:
    import unsloth
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import os
import sys
import json
import tempfile
import logging
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import gradio as gr
import requests
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image

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

INFERENCE_CONFIG = {
    "hf_model_repo": os.environ.get("HF_MODEL_REPO", "emadahmed97/matn-ocr-arabic-finetuned"),
    "base_model": "unsloth/DeepSeek-OCR",
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": True,
    "prompt": "<image>\nFree OCR. ",
}

class ArabicOCRTrainingSpace:
    """HuggingFace Spaces training and inference interface for Arabic OCR."""

    def __init__(self):
        self.training_active = False
        self.current_run_id = None
        self._inference_model = None
        self._inference_tokenizer = None
        self._model_loaded = False
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

            # _execute_training is a generator (yields progress updates)
            # Consume it fully and use the last yielded value as the result
            result = None
            for result in self._execute_training(full_config):
                pass

            return {
                "success": True,
                "message": result or "Training completed",
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

    def load_inference_model(self):
        """Load the fine-tuned LoRA model from HuggingFace Hub. Caches after first load."""
        if self._model_loaded:
            return self._inference_model, self._inference_tokenizer

        from pipelines.arabic_ocr.model import load_trained_model

        repo = INFERENCE_CONFIG["hf_model_repo"]
        logger.info(f"Loading inference model from {repo}...")

        model, tokenizer = load_trained_model(model_path=repo)
        self._inference_model = model
        self._inference_tokenizer = tokenizer
        self._model_loaded = True
        logger.info("Inference model loaded and cached.")
        return model, tokenizer

    def run_inference(self, image: Image.Image) -> str:
        """
        Run OCR inference on a single image.

        Args:
            image: PIL Image to process

        Returns:
            Extracted Arabic text
        """
        import io as _io
        import sys as _sys

        model, tokenizer = self.load_inference_model()

        # Save image to a temp file since model.infer() expects a file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            # model.infer() prints the OCR text to stdout and returns None,
            # so we capture stdout to get the actual text.
            capture = _io.StringIO()
            old_stdout = _sys.stdout
            _sys.stdout = capture

            try:
                result = model.infer(
                    tokenizer,
                    prompt=INFERENCE_CONFIG["prompt"],
                    image_file=tmp_path,
                    output_path=tempfile.gettempdir(),
                    base_size=INFERENCE_CONFIG["base_size"],
                    image_size=INFERENCE_CONFIG["image_size"],
                    crop_mode=INFERENCE_CONFIG["crop_mode"],
                    save_results=False,
                    test_compress=False,
                )
            finally:
                _sys.stdout = old_stdout

            # If infer() returned a string, use it directly
            if result is not None and str(result).strip():
                return str(result).strip()

            # Otherwise parse the captured stdout for the OCR text
            # The output contains debug lines like "BASE: ...", "PATCHES: ..."
            # followed by the actual OCR text, then "===save results===" etc.
            captured = capture.getvalue()
            lines = captured.strip().split("\n")
            text_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip debug/status lines
                if not stripped:
                    continue
                if stripped.startswith(("=", "BASE:", "PATCHES:", "image:", "other:")):
                    continue
                text_lines.append(stripped)

            return "\n".join(text_lines)
        finally:
            os.unlink(tmp_path)

    def gradio_infer(self, image) -> str:
        """Gradio handler for inference tab."""
        if image is None:
            return "Please upload an image."

        try:
            start = time.time()

            # Gradio Image component returns a numpy array or PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            text = self.run_inference(image)
            elapsed = time.time() - start

            return f"{text}\n\n---\nProcessed in {elapsed:.2f}s"

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return f"Inference failed: {str(e)}"

    def _execute_training(self, config: Dict[str, Any]) -> str:
        """Execute the actual training with the given configuration."""
        self.training_active = True
        self.current_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Import training pipeline
            from pipelines.arabic_ocr_training_pipeline import ArabicOCRTrainer

            # Try to import GPU_AVAILABLE, fallback if not available
            try:
                from pipelines.arabic_ocr_training_pipeline import GPU_AVAILABLE
            except ImportError:
                GPU_AVAILABLE = False

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
    """Create the Gradio interface with Training and Inference tabs."""

    with gr.Blocks(title="Arabic OCR - Training & Inference", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Arabic OCR - Matn")
        gr.Markdown("Fine-tune and run DeepSeek-OCR for Arabic manuscripts using LoRA")

        with gr.Tabs():
            # ── Inference Tab ──
            with gr.TabItem("Inference"):
                gr.Markdown("## Upload an image to extract Arabic text")
                gr.Markdown(
                    f"Model: `{INFERENCE_CONFIG['hf_model_repo']}` "
                    f"| Base: `{INFERENCE_CONFIG['base_model']}`"
                )

                with gr.Row():
                    with gr.Column():
                        infer_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        infer_btn = gr.Button("Run OCR", variant="primary", size="lg")

                    with gr.Column():
                        infer_output = gr.Textbox(
                            label="OCR Output",
                            lines=15,
                            rtl=True,
                            info="Extracted Arabic text (right-to-left)",
                        )

                infer_btn.click(
                    fn=training_space.gradio_infer,
                    inputs=[infer_image],
                    outputs=infer_output,
                )

                gr.Markdown("""
                **Tips**
                - First inference will take longer as the model loads (~30s)
                - Subsequent inferences reuse the cached model
                - Supports PNG, JPEG, and other common image formats
                """)

            # ── Training Tab ──
            with gr.TabItem("Training"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Training Configuration")

                        num_samples = gr.Slider(
                            minimum=10, maximum=10000, value=1000, step=10,
                            label="Number of Training Samples",
                            info="More samples = better quality but longer training"
                        )

                        max_steps = gr.Slider(
                            minimum=10, maximum=500, value=60, step=10,
                            label="Training Steps",
                            info="60 steps = 10 minutes. More steps = better convergence"
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
                        gr.Markdown("## Training Progress")

                        output_text = gr.Textbox(
                            lines=20,
                            label="Training Output",
                            info="Real-time training progress and results"
                        )

                with gr.Row():
                    train_btn = gr.Button("Start Training", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop Training", variant="secondary", size="lg")

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
                **Tips**
                - **Quick test**: 100 samples, 10 steps (~2 minutes)
                - **Development**: 1000 samples, 60 steps (~10 minutes)
                - **Production**: 5000+ samples, 200+ steps (~30+ minutes)
                - **Cost**: L4 GPU ~ $0.60/hour
                """)

    return interface


# Create the Gradio interface
demo = create_gradio_interface()

@demo.app.post("/api/train")
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

@demo.app.post("/api/infer")
async def infer_api(request: Request):
    """
    REST API endpoint for OCR inference.

    Accepts JSON with either:
      - "image_base64": base64-encoded image string
      - "image_url": URL to fetch the image from

    Returns JSON with "text" (the OCR result) and "processing_time_s".
    """
    try:
        data = await request.json()

        image_b64 = data.get("image_base64")
        image_url = data.get("image_url")

        if image_b64:
            import io
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif image_url:
            import io
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Provide 'image_base64' or 'image_url' in the request body."}
            )

        start = time.time()
        text = training_space.run_inference(image)
        elapsed = time.time() - start

        return JSONResponse(content={
            "text": text,
            "processing_time_s": round(elapsed, 3),
            "model": INFERENCE_CONFIG["hf_model_repo"],
        })

    except Exception as e:
        logger.error(f"API inference failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,
        show_error=True,
        ssr_mode=False,
    )
