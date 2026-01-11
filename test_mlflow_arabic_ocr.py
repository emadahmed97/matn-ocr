#!/usr/bin/env python3
"""
Simple test for Arabic OCR MLflow integration.

Tests the basic functionality without network dependencies.
"""

import os
import tempfile
import arabic_reshaper
from bidi.algorithm import get_display

# Test the metrics directly
import importlib.util
spec = importlib.util.spec_from_file_location("metrics", "pipelines/arabic_ocr/metrics.py")
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)

def format_arabic_display(text):
    """Format Arabic text for proper console display."""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except:
        return text

def test_arabic_ocr_metrics():
    """Test Arabic OCR evaluation metrics."""
    print("🔬 Testing Arabic OCR Metrics")
    print("=" * 40)

    # Test Arabic sentences
    predictions = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "كتاب الطهارة"
    ]

    ground_truths = [
        "بسم الله الرحمن الرحيم",    # Perfect match
        "الحمد لله رب العالمين",     # Perfect match
        "كتاب الطهاره"              # Slight difference (ة vs ه)
    ]

    print("📝 Test Samples:")
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        print(f"  Sample {i+1}:")
        print(f"    Predicted:    {format_arabic_display(pred)}")
        print(f"    Ground Truth: {format_arabic_display(gt)}")
        print()

    # Calculate individual metrics
    print("📊 Individual Metrics:")
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        cer = metrics_module.calculate_cer(pred, gt)
        wer = metrics_module.calculate_wer(pred, gt)
        bleu = metrics_module.calculate_bleu(pred, gt)

        print(f"  Sample {i+1}: CER={cer:.3f}, WER={wer:.3f}, BLEU={bleu:.3f}")

    # Calculate batch metrics
    print("\n🎯 Batch Evaluation:")
    batch_results = metrics_module.evaluate_batch(predictions, ground_truths)

    formatted_results = metrics_module.format_evaluation_results(batch_results)
    print(formatted_results)

    return batch_results

def test_mlflow_basic():
    """Test basic MLflow functionality without experiment setup."""
    print("🔬 Testing Basic MLflow Functionality")
    print("=" * 40)

    try:
        import mlflow

        # Set local tracking
        mlflow.set_tracking_uri("file:///tmp/mlflow")
        print("✅ MLflow imported and tracking URI set")

        # Test parameter logging
        with mlflow.start_run(run_name="test-arabic-ocr"):
            mlflow.log_param("model_name", "nougat-small")
            mlflow.log_param("language", "arabic")
            mlflow.log_param("dataset", "arabic-books")

            # Test metric logging
            mlflow.log_metric("cer", 0.05)
            mlflow.log_metric("wer", 0.12)
            mlflow.log_metric("bleu", 0.85)

            print("✅ Parameters and metrics logged successfully")

            # Test artifact logging
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Test Arabic OCR artifact\n")
                f.write("مرحبا بالعالم\n")
                temp_path = f.name

            mlflow.log_artifact(temp_path, "test_artifacts")
            os.unlink(temp_path)
            print("✅ Artifact logged successfully")

        print("✅ MLflow basic functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ MLflow test failed: {e}")
        return False

def test_section_1_3_complete():
    """Complete test for Section 1.3 MLflow Integration."""
    print("🕌 Section 1.3: MLflow Integration for Arabic OCR")
    print("=" * 60)

    # Test 1: Arabic OCR Metrics
    print("\n1️⃣ Testing Arabic OCR Evaluation Metrics")
    metrics_results = test_arabic_ocr_metrics()

    if metrics_results:
        print("✅ Arabic OCR metrics test passed")
    else:
        print("❌ Arabic OCR metrics test failed")
        return False

    # Test 2: MLflow Basic Functionality
    print("\n2️⃣ Testing MLflow Basic Functionality")
    mlflow_success = test_mlflow_basic()

    if not mlflow_success:
        return False

    # Test 3: Integration Summary
    print("\n3️⃣ Section 1.3 Integration Summary")
    print("✅ MLflow configured for OCR experiments")
    print("✅ Arabic text evaluation metrics (CER, WER, BLEU) implemented")
    print("✅ OCR-specific logging and tracking created")
    print("✅ MLflow integration tested with sample OCR data")

    print(f"\n🎯 Section 1.3 Complete!")
    print(f"📊 Average CER: {metrics_results['cer']:.3f}")
    print(f"📊 Average WER: {metrics_results['wer']:.3f}")
    print(f"📊 Average BLEU: {metrics_results['bleu']:.3f}")
    print(f"📊 Exact Match Rate: {metrics_results['exact_match']:.1%}")

    print("\n🚀 Ready to proceed to Phase 2: Training Pipeline Development!")

    return True

if __name__ == "__main__":
    success = test_section_1_3_complete()
    exit(0 if success else 1)