"""
DeepSeek-OCR model loading and configuration.

Extracted from notebooks/arabic_ocr_finetune.ipynb for production use.
"""

import os
from typing import Tuple, Optional
import torch
from unsloth import FastVisionModel
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download


def download_deepseek_ocr(local_dir: str = "deepseek_ocr") -> str:
    """
    Download DeepSeek-OCR model from HuggingFace.

    Args:
        local_dir: Directory to save the model

    Returns:
        Path to the downloaded model
    """
    model_path = snapshot_download("unsloth/DeepSeek-OCR", local_dir=local_dir)
    return model_path


def load_deepseek_ocr_model(
    model_path: str = "./deepseek_ocr",
    load_in_4bit: bool = False,
    use_gradient_checkpointing: bool = True,
    trust_remote_code: bool = True,
    download_if_missing: bool = True
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load DeepSeek-OCR model and tokenizer for training or inference.

    Args:
        model_path: Path to the model directory
        load_in_4bit: Whether to load in 4-bit quantization
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        trust_remote_code: Trust remote code execution
        download_if_missing: Download model if not found locally

    Returns:
        Tuple of (model, tokenizer)
    """
    # Set environment variable to suppress warnings
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

    # Download model if missing and requested
    if download_if_missing and not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        model_path = download_deepseek_ocr(model_path)

    try:
        # Load model and tokenizer using Unsloth FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            auto_model=AutoModel,
            trust_remote_code=trust_remote_code,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth" if use_gradient_checkpointing else False,
        )

        print(f"✅ Successfully loaded DeepSeek-OCR from {model_path}")
        print(f"📊 Model parameters: {model.num_parameters():,}")

        # Check device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")

        model.to(device)

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading DeepSeek-OCR: {e}")
        raise


def setup_lora_model(
    model: torch.nn.Module,
    target_modules: Optional[list] = None,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    bias: str = "none",
    use_rslora: bool = False,
    random_state: int = 3407
) -> torch.nn.Module:
    """
    Add LoRA adapters to the model for parameter-efficient fine-tuning.

    Args:
        model: The base DeepSeek-OCR model
        target_modules: List of modules to apply LoRA to
        r: LoRA rank (higher = more capacity but potential overfitting)
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate for LoRA layers
        bias: Bias handling strategy
        use_rslora: Use rank-stabilized LoRA
        random_state: Random seed

    Returns:
        Model with LoRA adapters added
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    try:
        peft_model = FastVisionModel.get_peft_model(
            model,
            target_modules=target_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=None,
        )

        # Count trainable parameters
        total_params = peft_model.num_parameters()
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        trainable_pct = (trainable_params / total_params) * 100

        print(f"🔧 LoRA setup complete!")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")

        return peft_model

    except Exception as e:
        print(f"❌ Error setting up LoRA: {e}")
        raise


def prepare_model_for_training(model: torch.nn.Module) -> torch.nn.Module:
    """
    Prepare the model for training mode.

    Args:
        model: The model to prepare

    Returns:
        Model ready for training
    """
    FastVisionModel.for_training(model)
    return model


def prepare_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Prepare the model for inference mode.

    Args:
        model: The model to prepare

    Returns:
        Model ready for inference
    """
    FastVisionModel.for_inference(model)
    return model


def save_model_and_tokenizer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    save_path: str = "lora_model",
    push_to_hub: bool = False,
    hub_model_name: Optional[str] = None,
    token: Optional[str] = None
) -> None:
    """
    Save model and tokenizer locally or to HuggingFace Hub.

    Args:
        model: The trained model
        tokenizer: The tokenizer
        save_path: Local path to save the model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_name: Name for the Hub model
        token: HuggingFace token for uploading
    """
    # Save locally
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Model saved locally to {save_path}")

    # Optionally push to hub
    if push_to_hub and hub_model_name:
        if not token:
            print("❌ Hub token required for uploading")
            return

        try:
            model.push_to_hub(hub_model_name, token=token)
            tokenizer.push_to_hub(hub_model_name, token=token)
            print(f"✅ Model uploaded to HuggingFace Hub: {hub_model_name}")
        except Exception as e:
            print(f"❌ Error uploading to Hub: {e}")


def load_trained_model(
    model_path: str,
    base_model_path: str = "./deepseek_ocr"
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a trained LoRA model for inference.

    Args:
        model_path: Path to the saved LoRA model
        base_model_path: Path to the base DeepSeek-OCR model

    Returns:
        Tuple of (model, tokenizer) ready for inference
    """
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )

        model = prepare_model_for_inference(model)
        print(f"✅ Trained model loaded from {model_path}")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        raise