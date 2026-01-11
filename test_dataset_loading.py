#!/usr/bin/env python3
"""
Test script for Arabic manuscript dataset loading.

This script tests the dataset pipeline and provides validation
for the image-text pair loading functionality.
"""

import sys
import os
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# Add the pipelines directory to path
sys.path.append('pipelines')
from arabic_ocr_dataset import ArabicManuscriptDataset, setup_dataset_directory

def create_sample_images(data_dir: Path, num_samples: int = 5):
    """Create sample manuscript-like images for testing."""
    images_dir = data_dir / "images"
    transcriptions_dir = data_dir / "transcriptions"

    # Sample Arabic texts from classical Islamic sources
    sample_texts = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "كتاب الطهارة - باب الوضوء",
        "قال رسول الله صلى الله عليه وسلم",
        "وعن أبي هريرة رضي الله عنه قال"
    ]

    created_files = []

    for i in range(min(num_samples, len(sample_texts))):
        text = sample_texts[i]

        # Create a simple image with Arabic text
        img = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(img)

        # Try to format Arabic text for display
        try:
            formatted_text = get_display(arabic_reshaper.reshape(text))
        except:
            formatted_text = text

        # Draw text (will be basic without proper Arabic font)
        try:
            draw.text((20, 80), formatted_text, fill='black')
        except:
            # Fallback for systems without Arabic font support
            draw.text((20, 80), f"Arabic Text {i+1}", fill='black')

        # Save image
        image_path = images_dir / f"sample_{i+1:03d}.jpg"
        img.save(image_path, quality=95)

        # Save corresponding transcription
        text_path = transcriptions_dir / f"sample_{i+1:03d}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)

        created_files.append((image_path, text_path, text))
        print(f"✅ Created sample {i+1}: {image_path.name}")

    return created_files

def test_dataset_loading():
    """Test the dataset loading functionality."""
    print("🔬 Testing Arabic Manuscript Dataset Loading")
    print("=" * 50)

    # Create temporary dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "test_manuscripts"

        # Set up directory structure
        setup_dataset_directory(str(data_dir))
        print(f"📁 Created test dataset at: {data_dir}")

        # Create sample images and transcriptions
        print("\n📝 Creating sample images and transcriptions...")
        sample_files = create_sample_images(data_dir, num_samples=3)

        # Test dataset loading
        print("\n📊 Testing dataset loading...")
        try:
            dataset = ArabicManuscriptDataset(str(data_dir), split="all")
            print(f"✅ Dataset loaded successfully: {len(dataset)} samples")

            # Test getting a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ Sample retrieved successfully")
                print(f"   - Image shape: {sample['pixel_values'].shape}")
                print(f"   - Text length: {len(sample['text'])} chars")
                print(f"   - Raw text: {format_arabic_display(sample['raw_text'])}")
                print(f"   - Processed text: {format_arabic_display(sample['text'])}")

                # Test statistics
                print("\n📈 Dataset statistics:")
                stats = dataset.get_statistics()
                for key, value in stats.items():
                    print(f"   - {key}: {value}")

            else:
                print("❌ No samples found in dataset")
                return False

        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            return False

    print("\n✅ Dataset loading test completed successfully!")
    return True

def format_arabic_display(text):
    """Format Arabic text for console display."""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except:
        return text

def test_csv_loading():
    """Test CSV-based dataset loading."""
    print("\n🔬 Testing CSV-based Dataset Loading")
    print("=" * 40)

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "csv_test"

        # Set up directory
        setup_dataset_directory(str(data_dir))

        # Create sample images
        create_sample_images(data_dir, num_samples=2)

        # Create CSV file
        csv_content = """image_path,text,split,source,page_number,book_title
images/sample_001.jpg,بسم الله الرحمن الرحيم,train,manuscript,1,كتاب التوحيد
images/sample_002.jpg,الحمد لله رب العالمين,val,manuscript,2,كتاب التوحيد"""

        csv_path = data_dir / "dataset.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)

        # Test loading
        try:
            dataset = ArabicManuscriptDataset(str(data_dir), split="all")
            print(f"✅ CSV dataset loaded: {len(dataset)} samples")

            # Test split filtering
            train_dataset = ArabicManuscriptDataset(str(data_dir), split="train")
            val_dataset = ArabicManuscriptDataset(str(data_dir), split="val")

            print(f"   - Train split: {len(train_dataset)} samples")
            print(f"   - Val split: {len(val_dataset)} samples")

            # Test metadata
            if len(dataset) > 0:
                sample = dataset[0]
                metadata = sample['metadata']
                print(f"   - Metadata: {metadata}")

        except Exception as e:
            print(f"❌ CSV dataset loading failed: {e}")
            return False

    print("✅ CSV dataset loading test completed!")
    return True

def main():
    """Main test function."""
    print("🕌 Section 2.1: Data Loading & Preprocessing Test Suite")
    print("=" * 60)

    success = True

    # Test 1: Basic dataset loading
    success &= test_dataset_loading()

    # Test 2: CSV-based loading
    success &= test_csv_loading()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All dataset loading tests passed!")
        print("\n📋 Ready for your manuscript images:")
        print("   📁 Place images in: data/manuscripts/images/")
        print("   📝 Place transcriptions in: data/manuscripts/transcriptions/")
        print("   📊 Or create dataset.csv with image paths and texts")
        print("\n🔢 Recommended amounts:")
        print("   🚀 Minimum: 10-20 images for testing")
        print("   🎯 Recommended: 50-100 images for training")
        print("   🏆 Optimal: 200+ images for production quality")
        print("\n📋 Image requirements:")
        print("   📷 Format: JPG, PNG, TIFF")
        print("   🔍 Resolution: 300+ DPI preferred")
        print("   📐 Size: Any reasonable size (will be resized to 896x672)")
        print("   🏷️ Naming: Sequential (e.g., page_001.jpg)")
    else:
        print("❌ Some tests failed!")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)