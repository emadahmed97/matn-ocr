# Matn-OCR: Arabic Manuscript OCR — Project Report

This report follows the ML Project Template structure, adapted to reflect the specific architecture and decisions made in the Matn-OCR project.

---

## 1. Problem Framing and Success Metrics

### Define the Problem

Classical Arabic manuscripts — Islamic texts, historical documents, and scholarly works — are largely inaccessible in digital form. Manual transcription is slow, expensive, and doesn't scale. The goal of Matn-OCR is to **automatically extract Arabic text from manuscript images** using a fine-tuned vision-language model.

ML is the right solution here because rule-based OCR systems struggle with Arabic's connected script, diacritics, and the visual complexity of handwritten/historical text. A fine-tuned deep learning model can learn these patterns from examples.

### User Story

**As a** researcher or student of classical Arabic texts,
**I want to** upload a photo or scan of a manuscript page,
**So that I** get accurate digital Arabic text I can search, copy, and study.

### ML Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **CER (Character Error Rate)** | Edit distance at the character level between predicted and ground truth text | < 5% (0.05) |
| **WER (Word Error Rate)** | Edit distance at the word level | < 10% |
| **BLEU Score** | N-gram overlap between prediction and reference | > 0.75 |
| **Exact Match** | Percentage of samples where prediction perfectly matches ground truth | Tracked, not gated on |

CER is the primary metric because Arabic OCR errors often manifest at the character level (missed diacritics, confused letter forms). The auto-deployment threshold is set at **CER < 0.05**.

### Business / Impact Metric

For this project, the "business metric" is accessibility: how many manuscript pages can be digitized per hour compared to manual transcription. A secondary metric is user satisfaction with OCR accuracy (measured qualitatively through the Gradio UI).

### Constraints

- **Latency**: First inference takes ~30s (model loading), subsequent inferences are faster due to caching. Acceptable for a research tool.
- **Cost**: Training runs on an L4 GPU via HuggingFace Spaces (~$0.60/hour). A typical training run (60 steps, 1000 samples) completes in ~10 minutes.
- **Hardware**: Inference and training both require GPU. The HF Space is configured with L4 hardware.
- **Data**: Arabic manuscript OCR datasets are limited. We use `mssqpi/Arabic-OCR-Dataset` (2.16M image-text pairs), which is one of the largest available.
- **Language complexity**: Arabic is written right-to-left, has connected letters that change form based on position, and includes optional diacritical marks. The metrics module includes Arabic-specific text normalization (alef variants, teh marbuta, tatweel removal).

---

## 2. Sourcing Unique Data

### Dataset

- **Source**: `mssqpi/Arabic-OCR-Dataset` on HuggingFace Hub
- **Size**: 2.16 million image-text pairs
- **Content**: Classical Arabic manuscript pages with corresponding transcriptions
- **Format**: Image + Arabic text ground truth

This is a pre-existing public dataset, so we didn't need to build a scraping pipeline or collect data manually. The dataset was chosen because it covers the specific domain (classical Islamic manuscripts) and is large enough for fine-tuning.

### Data Access Pattern

The dataset is loaded on-demand via the HuggingFace `datasets` library during training. For quick experiments, we subsample (default: 1000 samples). For production runs, 5000+ samples are used.

```python
dataset = load_dataset("mssqpi/Arabic-OCR-Dataset", split="train")
dataset = dataset.select(range(num_samples))
```

We did not implement custom data collection since the existing dataset is sufficient for the current scope. If the project expanded to cover modern Arabic documents or specific manuscript styles, targeted data collection would be needed.

---

## 3. Continuous Data Collection

This step is **partially applicable** to our project. Since we're fine-tuning on an existing static dataset, there's no continuous data ingestion pipeline. However, the architecture supports iterative improvement:

- **Re-training on demand**: The GitHub Actions workflow and `/api/train` endpoint allow triggering new training runs with different parameters at any time.
- **New data integration**: The training pipeline accepts any HuggingFace dataset in the same format, so swapping or augmenting the dataset is straightforward.
- **Future direction**: A feedback loop where users correct OCR outputs in the Gradio UI and those corrections feed back into the training set is a natural next step but is not yet implemented.

### Data Quality Checks

The `test_basic_env.py` script validates the environment before training begins, including checking that required files exist and dependencies are importable. The data collator (`DeepSeekOCRDataCollator`) handles malformed samples gracefully by skipping them rather than crashing.

---

## 4. Data Storage

### Model Weights and Artifacts

- **Base model**: Stored on HuggingFace Hub (`unsloth/DeepSeek-OCR`), downloaded on demand
- **Fine-tuned LoRA adapters**: Pushed to HuggingFace Hub (`emadahmed97/matn-ocr-arabic-finetuned`)
- **Training artifacts**: Saved locally to `outputs/{run_id}/` during training

### Training Data

- **Source**: Streamed from HuggingFace Hub — no local storage needed
- **Cache**: Temporarily cached in `/tmp/hf_cache` during training (configured via `HF_HOME` env var)

### Experiment Tracking Data

- **MLflow**: Configured with SQLite backend (`sqlite:///mlflow.db`) for local tracking
- **Weights & Biases**: Used as an alternative/complementary tracker via the Metaflow pipeline

### Data Versioning

We don't use DVC or similar tools since the training dataset is versioned on HuggingFace Hub. The `num_samples` parameter and dataset name are logged with every training run, providing reproducibility. Each training run's configuration is captured in a JSON record:

```json
{
  "run_id": "run_20260209_120000",
  "config": {
    "num_samples": 1000,
    "max_steps": 60,
    "model_name": "unsloth/DeepSeek-OCR",
    "dataset_name": "mssqpi/Arabic-OCR-Dataset"
  }
}
```

---

## 5. Feature Engineering

Traditional feature engineering (scaling, encoding, feature selection) doesn't directly apply to this project since we're using a vision-language model that operates on raw images and produces text. However, the analogous steps are:

### Image Preprocessing (`pipelines/arabic_ocr/preprocessing.py`)

The `ArabicImageProcessor` class handles all image preparation:

- **Enhancement**: Grayscale conversion, contrast boost (1.2x), sharpening via UnsharpMask — improves text clarity in degraded manuscripts
- **Resizing**: Images are resized with aspect-ratio-preserving padding to target dimensions
- **Global/local views**: The data collator creates a global view (1024x1024) for context and local patches (640x640) for detail, mimicking how humans read manuscripts

### Data Augmentation

Three augmentation levels are available to improve generalization:

| Level | Rotation | Brightness | Contrast | Other |
|-------|----------|-----------|----------|-------|
| Light | +/-2 deg | 0.9-1.1 | - | - |
| Medium | +/-5 deg | 0.8-1.2 | 0.9-1.1 | - |
| Heavy | +/-10 deg | 0.7-1.3 | 0.8-1.2 | Sharpness, blur |

### Text Normalization (for evaluation)

Arabic text is normalized before metric computation to avoid penalizing cosmetic differences:
- Alef variants (hamza forms) normalized to bare alef
- Alef maksura normalized to ya
- Teh marbuta normalized to ha
- Tatweel (kashida) removed
- Whitespace normalized

### Avoiding Data Leakage

The model is evaluated on held-out samples not seen during training. The `train_test_split` ratio is 0.8 in the Metaflow pipeline. During step-limited training (e.g., 60 steps), only a subset of the training data is seen, providing implicit separation.

---

## 6. Labeling

The labeling step is **already handled** by the source dataset. `mssqpi/Arabic-OCR-Dataset` contains pre-labeled image-text pairs where the text is the ground truth transcription of each manuscript image.

### Conversation Format

For the vision-language model, each sample is converted into a conversation format at training time:

```python
{
    "messages": [
        {
            "role": "<|User|>",
            "content": "<image>\nFree OCR. ",
            "images": [image_tensor]
        },
        {
            "role": "<|Assistant|>",
            "content": "بسم الله الرحمن الرحيم..."
        }
    ]
}
```

The data collator masks the user prompt during training so the model only learns to predict the assistant's response (the Arabic text), not to reproduce the prompt itself. This is controlled by `train_on_responses_only=True`.

### Label Quality

Since we rely on the upstream dataset's labels, label quality depends on the original annotators. The evaluation metrics (CER, WER) give us a signal on whether the model is learning meaningful patterns from the labels. If label noise were a concern, we could implement sample-level CER analysis to identify and filter noisy examples.

---

## 7. Model Training and Evaluation

### Model Architecture

- **Base model**: `unsloth/DeepSeek-OCR` — a vision-language model specialized for OCR tasks
- **Architecture**: Vision encoder (processes image patches) + language decoder (generates text)
- **Fine-tuning method**: LoRA (Low-Rank Adaptation) via Unsloth

### LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Balance between capacity and efficiency |
| Alpha | 16 | Standard choice (alpha = r) |
| Dropout | 0 | Unsloth recommends 0 for LoRA |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + MLP projections |

LoRA keeps the base model frozen and trains small adapter matrices, making fine-tuning feasible on a single L4 GPU with ~1-5M trainable parameters.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 2 (per device) |
| Gradient accumulation | 4 (effective batch: 8) |
| Learning rate | 2e-4 |
| Scheduler | Linear decay |
| Warmup steps | 5 |
| Max steps | 60 (quick) / 200+ (production) |
| Optimizer | AdamW 8-bit |
| Precision | fp16 or bf16 (hardware-dependent) |
| Weight decay | 0.001 |
| Seed | 3407 |

### Training Tiers

| Tier | Samples | Steps | Time | Use Case |
|------|---------|-------|------|----------|
| Quick test | 100 | 10 | ~2 min | Sanity check |
| Development | 1,000 | 60 | ~10 min | Iteration |
| Production | 5,000+ | 200+ | ~30+ min | Deployment |

### Experiment Tracking

Two systems are integrated:

**MLflow** (`mlflow_arabic_ocr_config.py`):
- Tracks dataset info, model config, per-epoch metrics, OCR evaluation, and sample predictions
- SQLite backend for local runs
- Logs CER, WER, BLEU, and exact match after training

**Weights & Biases** (via Metaflow pipeline):
- Used in the `pipelines/arabic_ocr/training.py` Metaflow flow
- Logs hyperparameters and training curves

### Model Versioning

- Trained LoRA adapters are pushed to HuggingFace Hub: `emadahmed97/matn-ocr-arabic-finetuned`
- Each training run is identified by a unique `run_id` (timestamp + git SHA)
- Local saves go to `outputs/{run_id}/` and `models/{run_id}/`

### Evaluation

The metrics module (`pipelines/arabic_ocr/metrics.py`) computes:

1. **CER** — primary metric, character-level edit distance
2. **WER** — word-level edit distance
3. **BLEU** — n-gram overlap (1-4 grams)
4. **Exact Match** — binary per-sample accuracy

Evaluation is run on held-out samples after training. The Metaflow pipeline evaluates on 50 samples; the app's training flow checks whether the final train loss meets the deployment threshold.

### Auto-Deployment Gate

If `train_loss < deploy_threshold` (default: 0.05), the model is automatically saved for deployment. This is a simple but effective gate — future iterations could use CER on a validation set instead of train loss.

---

## 8. Deployment

### Interactive App (Gradio)

The primary interface is a **Gradio 5 web application** hosted on HuggingFace Spaces with two tabs:

**Inference Tab:**
- Upload a manuscript image (PNG, JPEG, clipboard)
- Click "Run OCR" to extract Arabic text
- Output displayed in RTL format with processing time

**Training Tab:**
- Configure training parameters (samples, steps, learning rate, threshold)
- Start training with real-time progress streaming
- Monitor training output live

### REST API (FastAPI)

Two API endpoints are registered on Gradio's internal FastAPI app:

**POST `/api/train`** — Trigger automated training
```json
// Request
{"num_samples": 1000, "max_steps": 60, "deploy_threshold": 0.05}

// Response
{"success": true, "message": "Training completed", "run_id": "run_...", "status": "completed"}
```

**POST `/api/infer`** — Run OCR inference
```json
// Request (base64)
{"image_base64": "iVBORw0KGgo..."}

// Request (URL)
{"image_url": "https://example.com/manuscript.png"}

// Response
{"text": "بسم الله الرحمن الرحيم", "processing_time_s": 2.345, "model": "emadahmed97/matn-ocr-arabic-finetuned"}
```

### Docker

A `Dockerfile` is provided for containerized deployment:

```dockerfile
FROM python:3.10-slim
# Install deps, copy code, expose port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### CI/CD (GitHub Actions)

Two workflows automate the deployment pipeline:

**1. Sync to HuggingFace Spaces** (`sync-to-hf-spaces.yml`)
- Triggers on every push to `main`
- Creates an orphan branch (no git history / large files)
- Force-pushes to `huggingface.co/spaces/emadahmed97/arabic-ocr-trainer`
- Uses the original commit message

**2. Arabic OCR Training Pipeline** (`arabic-ocr-training.yml`)
- Triggers on push to `pipelines/`, `requirements.txt`, `app.py`, or the workflow file
- Also supports manual trigger with configurable parameters
- **4 jobs**:
  1. **Validate** — install deps, run `test_basic_env.py`
  2. **Train** — POST to the HF Space's `/api/train` endpoint to trigger GPU training
  3. **Monitor** — placeholder for polling training status
  4. **Notify** — report pipeline success/failure

This architecture keeps GPU-intensive work on HuggingFace Spaces (which has the L4 GPU) while using GitHub Actions (CPU-only) for orchestration.

### Tests

- **`test_basic_env.py`**: Validates dependencies, file structure, Python syntax, requirements.txt format, and README frontmatter. Used in CI.
- **`test_imports.py`**: Comprehensive dependency import test including GPU/CUDA checks and Unsloth detection.

---

## 9. Monitoring

Monitoring is **partially implemented** in the current version. Here's what exists and what's planned:

### What's Implemented

- **Training progress streaming**: The Gradio training tab shows real-time progress (model loading, dataset loading, training steps, final metrics)
- **Experiment logging**: MLflow and W&B log training metrics, dataset info, and model configs for every run
- **API response logging**: FastAPI logs all requests via Python's `logging` module
- **Training run records**: Each CI-triggered training creates a JSON record with config, timestamp, and status

### What's Planned (TODOs in code)

- **Training status polling**: The `monitor` job in the GitHub Actions workflow has a TODO to poll the HF Spaces API for training status
- **Prediction logging**: Individual inference requests are not yet logged with timestamps and inputs for drift detection
- **Input data distribution monitoring**: No automated checks for unexpected image sizes, formats, or text distributions
- **Feedback loops**: No mechanism yet for users to correct OCR outputs and feed corrections back into training

### Re-training Triggers

Currently, re-training is triggered in two ways:
1. **Automatically**: When code changes are pushed to `pipelines/` or related files
2. **Manually**: Via the GitHub Actions `workflow_dispatch` or the Gradio training tab

A more sophisticated approach would monitor CER on incoming production images and trigger re-training when performance degrades — this is a future improvement.

---

## Summary

Matn-OCR is a focused, end-to-end ML project that fine-tunes a vision-language model for Arabic manuscript OCR. It covers most steps of the ML project lifecycle:

| Step | Status | Notes |
|------|--------|-------|
| 1. Problem Framing | Done | CER < 5% target, clear user story |
| 2. Data Sourcing | Done | 2.16M pairs from HuggingFace |
| 3. Continuous Collection | Partial | Manual re-training supported, no automated collection |
| 4. Data Storage | Done | HF Hub for models and data, MLflow for experiments |
| 5. Feature Engineering | Done | Image preprocessing, augmentation, Arabic normalization |
| 6. Labeling | Done | Pre-labeled dataset, conversation format conversion |
| 7. Training & Eval | Done | LoRA fine-tuning, CER/WER/BLEU metrics, experiment tracking |
| 8. Deployment | Done | Gradio UI, REST API, Docker, CI/CD with GitHub Actions |
| 9. Monitoring | Partial | Training logging exists, production monitoring is future work |

The project prioritizes simplicity and iteration speed — a single L4 GPU can train a model in 10 minutes, and the CI/CD pipeline automatically deploys code changes to HuggingFace Spaces.
