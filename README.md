---
title: Matn - Arabic OCR
emoji: 🕌
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.1.0
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
hardware: l4
datasets:
- mssqpi/Arabic-OCR-Dataset
models:
- unsloth/DeepSeek-OCR
tags:
- arabic
- ocr
- computer-vision
- training
- lora
---

# Matn - Arabic OCR for Classical Islamic Texts

An end-to-end machine learning system for extracting text from classical Arabic Islamic manuscript images, powered by DeepSeek-OCR with LoRA fine-tuning via Unsloth.

## Features

- **OCR Inference**: Upload a manuscript image and get Arabic text output (RTL formatted)
- **Training Pipeline**: Fine-tune DeepSeek-OCR with LoRA on custom Arabic datasets
- **REST API**: `/api/infer` for programmatic OCR, `/api/train` for automated training
- **MLflow Tracking**: Experiment tracking with CER/WER/BLEU metrics
- **GitHub Actions**: Automated training triggers on code changes, auto-sync to HF Spaces

## Architecture

```
Input: Manuscript Image (PNG/JPEG)
  |
DeepSeek-OCR Vision Encoder
  |
Language Model Decoder (with LoRA adapters)
  |
Output: Arabic Text
```

**Dataset**: [mssqpi/Arabic-OCR-Dataset](https://huggingface.co/datasets/mssqpi/Arabic-OCR-Dataset) (2.16M image-text pairs)
**Trained Model**: [emadahmed97/matn-ocr-arabic-finetuned](https://huggingface.co/emadahmed97/matn-ocr-arabic-finetuned)

## Usage

### Inference (Gradio UI)

Visit the HuggingFace Space and use the **Inference** tab to upload an image and get OCR results.

### Inference (API)

```python
import requests, base64

with open("manuscript.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://emadahmed97-arabic-ocr-trainer.hf.space/api/infer",
    json={"image_base64": image_b64}
)
print(response.json()["text"])
```

### Training

Training can be triggered via:
1. **Gradio UI**: Use the Training tab on HF Spaces
2. **REST API**: `POST /api/train` with training config
3. **GitHub Actions**: Push changes to `pipelines/` to auto-trigger

## Development

```bash
# Clone and set up
git clone https://github.com/emadahmed97/matn-ocr.git
cd matn-ocr

# Install dependencies
pip install -r requirements.txt

# Run basic environment tests
python test_basic_env.py
```

## Project Structure

```
app.py                          # Main Gradio + FastAPI application
requirements.txt                # Python dependencies
mlflow_arabic_ocr_config.py     # MLflow configuration
pipelines/
  arabic_ocr/
    model.py                    # Model loading (base + LoRA)
    preprocessing.py            # Image preprocessing
    data_collator.py            # Training data collation
    metrics.py                  # CER/WER/BLEU evaluation
  arabic_ocr_training_pipeline.py  # Training orchestration
notebooks/                      # Reference notebooks
.github/workflows/
  sync-to-hf-spaces.yml        # Auto-sync to HF Spaces
  arabic-ocr-training.yml      # Automated training pipeline
```

## License

Apache 2.0
