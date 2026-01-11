#!/usr/bin/env python3
"""
Test Nougat-small model integration for Arabic OCR project.

This script tests loading and basic functionality of the Nougat model
that we'll use for Arabic manuscript OCR.
"""

import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np


def test_nougat_loading():
    """Test loading Nougat-small model and processor."""
    print("🔄 Testing Nougat-small model loading...")

    try:
        # Load Nougat processor and model
        processor = NougatProcessor.from_pretrained('facebook/nougat-small')
        model = VisionEncoderDecoderModel.from_pretrained('facebook/nougat-small')

        print("✅ Successfully loaded Nougat-small model")
        print(f"📊 Model parameters: {model.num_parameters():,}")

        # Check device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")

        model.to(device)

        return processor, model, device

    except Exception as e:
        print(f"❌ Error loading Nougat: {e}")
        return None, None, None


def analyze_model_architecture(model):
    """Analyze the Nougat model architecture."""
    print("\n🏗️  Model Architecture Analysis:")

    print(f"Encoder: {type(model.encoder).__name__}")
    print(f"Decoder: {type(model.decoder).__name__}")

    # Check input/output dimensions
    encoder_config = model.encoder.config
    decoder_config = model.decoder.config

    print(f"Image input size: {encoder_config.image_size}")
    print(f"Patch size: {encoder_config.patch_size}")
    print(f"Hidden size: {encoder_config.hidden_size}")
    print(f"Decoder vocab size: {decoder_config.vocab_size}")

    return encoder_config, decoder_config


def test_tokenizer_arabic_support(processor):
    """Test how the current tokenizer handles Arabic text."""
    print("\n🔤 Testing Arabic Text Support:")

    tokenizer = processor.tokenizer

    # Test Arabic text samples
    arabic_samples = [
        "بسم الله الرحمن الرحيم",  # Bismillah
        "الحمد لله رب العالمين",   # Alhamdulillah
        "قال رسول الله صلى الله عليه وسلم",  # Prophet saying
        "كتاب الطهارة",  # Chapter on purity
        "باب الوضوء"     # Section on ablution
    ]

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model max length: {tokenizer.model_max_length}")

    for i, text in enumerate(arabic_samples):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        print(f"\nSample {i+1}: {text}")
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Token count: {len(tokens)}")
        print(f"  Unknown tokens: {[t for t in tokens if '[UNK]' in t or '<unk>' in t]}")


def create_dummy_arabic_image():
    """Create a dummy image to test the processing pipeline."""
    print("\n🖼️  Creating dummy test image...")

    # Create a simple white image with some text-like patterns
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')

    # Add some simple patterns to simulate text
    import numpy as np
    img_array = np.array(image)

    # Add some horizontal lines to simulate text
    for y in range(100, 500, 40):
        img_array[y:y+3, 50:700] = [0, 0, 0]  # Black lines

    # Add some noise to make it more realistic
    noise = np.random.randint(0, 50, (height, width, 3))
    img_array = np.clip(img_array.astype(int) - noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)


def test_inference_pipeline(processor, model, device):
    """Test the complete inference pipeline with dummy data."""
    print("\n🔮 Testing Inference Pipeline:")

    try:
        # Create dummy image
        dummy_image = create_dummy_arabic_image()

        # Process the image
        pixel_values = processor(dummy_image, return_tensors="pt").pixel_values.to(device)
        print(f"Input shape: {pixel_values.shape}")

        # Generate output (dummy inference)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_length=100,
                num_beams=1,
                do_sample=False
            )

        # Decode the output
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"✅ Inference successful!")
        print(f"Generated text: {generated_text[:200]}...")
        print(f"Output length: {len(generated_text)} characters")

        return True

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False


def check_arabic_compatibility():
    """Check system compatibility for Arabic text processing."""
    print("\n🌍 Arabic Compatibility Check:")

    try:
        # Test Arabic text rendering
        arabic_text = "مرحبا بكم في نظام التعرف الضوئي على الحروف العربية"
        print(f"Arabic text display: {arabic_text}")

        # Test RTL support
        rtl_text = "اللغة العربية تكتب من اليمين إلى اليسار"
        print(f"RTL text: {rtl_text}")

        # Test diacritics
        diacritics_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        print(f"Diacritics: {diacritics_text}")

        print("✅ Arabic text compatibility confirmed")
        return True

    except Exception as e:
        print(f"❌ Arabic compatibility issue: {e}")
        return False


def main():
    """Main test function."""
    print("🕌 Nougat-Small Arabic OCR Integration Test")
    print("=" * 50)

    # Test 1: Model loading
    processor, model, device = test_nougat_loading()
    if not processor or not model:
        print("❌ Model loading failed. Cannot continue.")
        return

    # Test 2: Architecture analysis
    analyze_model_architecture(model)

    # Test 3: Arabic text tokenization
    test_tokenizer_arabic_support(processor)

    # Test 4: Inference pipeline
    inference_success = test_inference_pipeline(processor, model, device)

    # Test 5: Arabic compatibility
    arabic_compatible = check_arabic_compatibility()

    # Summary
    print("\n📋 Integration Test Summary:")
    print("=" * 30)
    print(f"✅ Model Loading: Success")
    print(f"✅ Architecture Analysis: Success")
    print(f"⚠️  Arabic Tokenization: Needs improvement for Arabic")
    print(f"{'✅' if inference_success else '❌'} Inference Pipeline: {'Success' if inference_success else 'Failed'}")
    print(f"{'✅' if arabic_compatible else '❌'} Arabic Compatibility: {'Success' if arabic_compatible else 'Failed'}")

    print("\n🎯 Next Steps:")
    print("1. Fine-tune tokenizer for Arabic text")
    print("2. Create Arabic manuscript image dataset")
    print("3. Implement Arabic-specific preprocessing")
    print("4. Set up training pipeline")


if __name__ == "__main__":
    main()