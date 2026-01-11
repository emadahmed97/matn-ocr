#!/usr/bin/env python3
"""
Script to explore the MohamedRashad/arabic-books dataset for our Nougat OCR project.

This dataset contains 8,647 Arabic books with ~4.8GB of text data.
Perfect for training Arabic OCR models on classical Islamic texts.
"""

from datasets import load_dataset_builder, load_dataset
import sys


def explore_dataset_info():
    """Get basic dataset information without downloading."""
    print("🔍 Exploring MohamedRashad/arabic-books dataset...")

    builder = load_dataset_builder('MohamedRashad/arabic-books')

    print(f"📚 Description: {builder.info.description or 'Arabic books collection'}")
    print(f"🗂️  Features: {builder.info.features}")
    print(f"📊 Splits: {builder.info.splits}")

    if 'train' in builder.info.splits:
        split_info = builder.info.splits['train']
        print(f"📖 Training examples: {split_info.num_examples:,}")
        print(f"💾 Dataset size: {split_info.num_bytes / (1024**3):.2f} GB")

    return builder


def load_sample_streaming():
    """Load a few samples using streaming to avoid full download."""
    print("\n🌊 Loading samples with streaming...")

    try:
        # Use streaming to avoid downloading everything
        ds = load_dataset('MohamedRashad/arabic-books', streaming=True)
        train_stream = iter(ds['train'])

        print("📄 First few samples:")
        for i, sample in enumerate(train_stream):
            if i >= 3:  # Just show first 3 samples
                break

            text = sample['text']
            print(f"\n--- Sample {i+1} ---")
            print(f"Length: {len(text):,} characters")
            print(f"First 200 chars: {text[:200]}...")

            # Check if it contains classical Arabic markers
            classical_markers = ['الله', 'صلى', 'رحمه', 'قال', 'كتاب']
            found_markers = [marker for marker in classical_markers if marker in text[:1000]]
            if found_markers:
                print(f"Classical markers found: {found_markers}")

        return True

    except Exception as e:
        print(f"❌ Error loading streaming dataset: {e}")
        return False


def check_suitability_for_ocr():
    """Analyze if this dataset is suitable for OCR training."""
    print("\n🎯 Dataset Analysis for Nougat OCR:")

    print("✅ Pros:")
    print("  - Large Arabic text corpus (8,647 books)")
    print("  - Classical/Islamic content (perfect for our use case)")
    print("  - Text-only format (good for ground truth)")
    print("  - Substantial size (4.8GB of text)")

    print("\n⚠️  Challenges:")
    print("  - Text-only (we need image-text pairs for OCR)")
    print("  - No manuscript images provided")
    print("  - Need to synthesize images or find complementary image dataset")

    print("\n💡 Next Steps:")
    print("  1. Use this as ground truth text corpus")
    print("  2. Generate synthetic manuscript images from this text")
    print("  3. Or pair with existing Arabic manuscript image datasets")
    print("  4. Create image-text pairs for Nougat fine-tuning")


def main():
    """Main exploration function."""
    print("🕌 Arabic Books Dataset Explorer for Nougat OCR")
    print("=" * 50)

    # Get dataset info
    builder = explore_dataset_info()

    # Try to load samples
    if len(sys.argv) > 1 and sys.argv[1] == "--stream":
        success = load_sample_streaming()
        if not success:
            print("⚠️  Could not load streaming samples. Dataset might be large or have connection issues.")
    else:
        print("\n💡 Run with --stream flag to load sample data")
        print("   python explore_arabic_dataset.py --stream")

    # Analysis
    check_suitability_for_ocr()

    print("\n🚀 Ready to proceed with Nougat integration!")


if __name__ == "__main__":
    main()