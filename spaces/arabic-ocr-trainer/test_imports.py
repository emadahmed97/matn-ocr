#!/usr/bin/env python3
"""
Test script to validate all imports work before deploying to HuggingFace Spaces.
Run this locally to catch missing dependencies early.
"""

import sys
import traceback
from pathlib import Path

def test_import(module_name, description=""):
    """Test importing a module and report result."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} - {description}: Unexpected error: {e}")
        return False

def test_core_dependencies():
    """Test core ML dependencies."""
    print("🔍 Testing Core Dependencies...")

    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "PyTorch Vision"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("accelerate", "HuggingFace Accelerate"),
        ("peft", "Parameter Efficient Fine-Tuning"),
        ("unsloth", "Unsloth (GPU acceleration)"),
        ("gradio", "Gradio UI Framework"),
        ("fastapi", "FastAPI Web Framework"),
        ("mlflow", "MLflow Experiment Tracking"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow Image Library"),
        ("requests", "HTTP Requests"),
        ("arabic_reshaper", "Arabic Text Reshaping"),
        ("bidi.algorithm", "Bidirectional Text"),
        ("wandb", "Weights & Biases"),
    ]

    failed = 0
    for module, desc in tests:
        if not test_import(module, desc):
            failed += 1

    return failed

def test_optional_dependencies():
    """Test optional/advanced dependencies."""
    print("\n🔍 Testing Optional Dependencies...")

    tests = [
        ("xformers", "Memory Efficient Transformers"),
        ("trl", "Transformer Reinforcement Learning"),
        ("bitsandbytes", "8-bit Quantization"),
        ("einops", "Tensor Operations"),
        ("safetensors", "Safe Tensor Format"),
        ("scipy", "Scientific Computing"),
        ("tokenizers", "Fast Tokenizers"),
        ("sentencepiece", "SentencePiece Tokenizer"),
        ("ninja", "Build System"),
        ("addict", "Dict Utilities"),
        ("easydict", "Easy Dict Access"),
        ("psutil", "System Utilities"),
        ("packaging", "Package Utilities"),
        ("protobuf", "Protocol Buffers"),
        ("regex", "Regular Expressions"),
    ]

    failed = 0
    for module, desc in tests:
        if not test_import(module, desc):
            failed += 1

    return failed

def test_custom_modules():
    """Test our custom modules."""
    print("\n🔍 Testing Custom Modules...")

    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    tests = [
        ("mlflow_arabic_ocr_config", "MLflow OCR Config"),
        ("pipelines.arabic_ocr.metrics", "OCR Metrics"),
        ("pipelines.arabic_ocr.preprocessing", "Image Preprocessing"),
        ("pipelines.arabic_ocr.data_collator", "Data Collator"),
        ("pipelines.arabic_ocr_training_pipeline", "Training Pipeline"),
    ]

    failed = 0
    for module, desc in tests:
        if not test_import(module, desc):
            failed += 1

    return failed

def test_gpu_availability():
    """Test GPU and CUDA availability."""
    print("\n🔍 Testing GPU/CUDA...")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️ No GPU detected (expected in CPU-only environment)")

        # Test Unsloth GPU detection
        try:
            from pipelines.arabic_ocr_training_pipeline import GPU_AVAILABLE
            print(f"✅ Unsloth GPU detection: {GPU_AVAILABLE}")
        except:
            print("⚠️ Could not import GPU_AVAILABLE from training pipeline")

    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return 1

    return 0

def test_app_startup():
    """Test that the main app can be imported."""
    print("\n🔍 Testing App Startup...")

    try:
        # This should import without errors
        from app import DEFAULT_CONFIG, ArabicOCRTrainingSpace
        print("✅ Main app imports successfully")
        print(f"✅ Default config loaded: {len(DEFAULT_CONFIG)} settings")

        # Test creating training space
        training_space = ArabicOCRTrainingSpace()
        print("✅ ArabicOCRTrainingSpace created successfully")

        return 0
    except Exception as e:
        print(f"❌ App startup failed: {e}")
        traceback.print_exc()
        return 1

def main():
    """Run all tests."""
    print("🧪 Testing Arabic OCR Training Environment")
    print("=" * 50)

    total_failed = 0

    # Test all dependency categories
    total_failed += test_core_dependencies()
    total_failed += test_optional_dependencies()
    total_failed += test_custom_modules()
    total_failed += test_gpu_availability()
    total_failed += test_app_startup()

    print("\n" + "=" * 50)
    if total_failed == 0:
        print("🎉 All tests passed! Environment is ready for deployment.")
        return 0
    else:
        print(f"❌ {total_failed} test(s) failed. Fix dependencies before deploying.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)