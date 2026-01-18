---
title: Arabic OCR Training
emoji: 🕌
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
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
duplicated_from: null
---

# Arabic OCR Training Space

This HuggingFace Space provides automated Arabic OCR model training using DeepSeek-OCR and LoRA fine-tuning.

## Features

- 🎯 **Gradio UI** for manual training with real-time progress
- 🔌 **REST API** for GitHub Actions automation
- ⚡ **L4 GPU** optimized for efficient training
- 📊 **MLflow tracking** for experiment management
- 🚀 **Auto-deployment** based on performance thresholds

## Training Configuration

- **Base Model**: unsloth/DeepSeek-OCR
- **Dataset**: mssqpi/Arabic-OCR-Dataset (2.16M samples)
- **Method**: LoRA fine-tuning (efficient, 2% parameters)
- **Hardware**: L4 GPU (24GB VRAM)
- **Cost**: ~$0.10 per development run (60 steps)

## Usage

### Manual Training (Gradio UI)
1. Adjust parameters (samples, steps, learning rate)
2. Click "Start Training"
3. Monitor real-time progress
4. Model auto-deploys if performance meets threshold

### Automated Training (API)
```bash
curl -X POST "https://huggingface.co/spaces/YOUR-ORG/arabic-ocr-trainer/api/train" \
  -H "Content-Type: application/json" \
  -d '{
    "num_samples": 1000,
    "max_steps": 60,
    "deploy_threshold": 0.05,
    "experiment_name": "auto-run"
  }'
```

## Performance Expectations

- **Training Time**: 60 steps ≈ 10 minutes on L4 GPU
- **Target CER**: < 5% for production deployment
- **Memory Usage**: ~14GB VRAM (fits L4 comfortably)
- **Cost Efficiency**: LoRA training vs full fine-tuning saves 90% compute

## Integration

This space integrates with:
- **GitHub Actions** for automated training triggers
- **MLflow** for experiment tracking and model versioning
- **HuggingFace Hub** for model storage and deployment

---

Built with the ML School Arabic OCR pipeline.