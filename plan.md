# Matn - Arabic OCR for Classical Islamic Texts

## Project Overview

We're building an end-to-end machine learning system that can:
- Extract text from classical Arabic Islamic manuscript images
- Provide structured output with proper formatting
- Handle diacritics and classical Arabic conventions
- Deploy as a production-ready service

**Dataset**: mssqpi/Arabic-OCR-Dataset (2.16M image-text pairs)
**Model**: DeepSeek-OCR (fine-tuned with LoRA via Unsloth)
**Architecture**: Vision Transformer Encoder → Language Model Decoder
**Trained Model**: https://huggingface.co/emadahmed97/matn-ocr-arabic-finetuned

## Implementation Phases

### Phase 1: Introduction & Setup

#### 1.1 Environment Setup
- ✅ Install required dependencies (datasets, transformers, unsloth)
- ✅ Explore Arabic OCR dataset structure
- ✅ Set up DeepSeek-OCR model integration
- ✅ Configure Arabic text processing pipeline

#### 1.2 Data Exploration & Analysis (EDA)
- ✅ Dataset statistics and sample analysis
- ✅ Arabic text characteristics analysis
- ✅ Classical Islamic text patterns identification
- ✅ Diacritics and formatting analysis

#### 1.3 MLflow Integration for Arabic OCR
- ✅ Configure MLflow for OCR experiments
- ✅ Set up Arabic text evaluation metrics
- ✅ Create OCR-specific logging and tracking

### Phase 2: Training Pipeline Development

#### 2.1 Data Loading & Preprocessing
- ✅ Load `mssqpi/Arabic-OCR-Dataset` via HuggingFace datasets
- ✅ Implement Arabic text normalization
- ✅ Convert dataset to conversation format for fine-tuning
- ✅ DeepSeekOCRDataCollator for image-text pair processing

#### 2.2 Model Architecture Setup
- ✅ DeepSeek-OCR as base vision-language model
- ✅ Configure LoRA fine-tuning for efficient training (2% of parameters)
- ✅ Set up Unsloth for 2x faster training
- ✅ Implement conversation-based training format

#### 2.3 Cross-Validation Strategy
- 🔲 Adapt cross-validation for OCR tasks
- 🔲 Implement text-based evaluation splits
- 🔲 Handle Arabic text-specific validation

#### 2.4 Training Implementation
- ✅ Fine-tune DeepSeek-OCR with LoRA adapters
- ✅ Implement production training pipeline with MLflow tracking
- ✅ Configure training hyperparameters for efficient fine-tuning
- ✅ Add conversation format data processing

#### 2.5 Evaluation Metrics
- ✅ Character Error Rate (CER)
- ✅ Word Error Rate (WER)
- ✅ BLEU score for text quality
- ✅ Diacritic accuracy assessment
- ✅ Islamic terminology recognition accuracy

#### 2.6 Model Registration
- ✅ Register best performing models (integrated in training pipeline)
- ✅ Version control for Arabic OCR models (via MLflow tracking)
- ✅ Model metadata and documentation (automated via pipeline)

### Phase 3: MLOps Automation Pipeline

#### 3.1 GitHub Actions Automation
- ✅ Create workflow for automated training triggers
- ✅ Set up data validation and testing pipeline
- ✅ Implement automated model performance gating
- ✅ Add single-environment deployment (direct to prod)

#### 3.2 HuggingFace Spaces Training Environment
- ✅ Set up GPU-enabled training space (L4 GPU)
- ✅ Create Gradio interface for manual training
- ✅ Implement REST API for automated training calls
- ✅ Add real-time training progress monitoring

#### 3.3 Model Registry & Versioning
- ✅ Auto-push trained LoRA adapters to HuggingFace Hub (via HF_TOKEN + HF_MODEL_REPO)
- ✅ Model saved at: https://huggingface.co/emadahmed97/matn-ocr-arabic-finetuned
- 🔲 A/B testing infrastructure setup
- 🔲 Model promotion workflow (dev → staging → prod)

#### 3.4 Inference Pipeline
- ✅ Add Inference tab to HF Spaces Gradio UI (upload image → OCR text output)
- ✅ Add `/api/infer` REST endpoint for programmatic inference
- ✅ Load LoRA model from HF Hub (`emadahmed97/matn-ocr-arabic-finetuned`)
- ✅ Handle RTL text formatting and confidence scoring

### Phase 4: Evaluation & Monitoring Pipeline
Comprehensive monitoring and evaluation system:

#### 4.1 Automated Evaluation Metrics
- 🔲 Real-time CER/WER/BLEU calculation during training
- 🔲 Arabic-specific metrics (diacritic accuracy, Islamic terminology)
- 🔲 Performance benchmarking against baseline models
- 🔲 Automated model comparison and ranking

#### 4.2 Production Model Monitoring
- 🔲 OCR accuracy tracking in production
- 🔲 Model drift detection (performance degradation)
- 🔲 Latency and throughput monitoring
- 🔲 Cost tracking (GPU usage, API calls)

#### 4.3 Data Quality Monitoring
- 🔲 Input image quality assessment
- 🔲 Arabic text output validation
- 🔲 Character distribution monitoring
- 🔲 Detection of adversarial or out-of-domain inputs

#### 4.4 MLOps Monitoring Dashboard
- 🔲 Training pipeline health and status
- 🔲 Model performance trends over time
- 🔲 A/B testing results visualization
- 🔲 Automated alerting for performance issues

#### 4.5 Continuous Evaluation & Testing
- 🔲 Automated testing pipeline with held-out datasets
- 🔲 Synthetic Arabic manuscript generation for testing
- 🔲 Human evaluation workflow integration
- 🔲 Automated retraining triggers based on performance

### Potential Future Work

#### Model Serving (Standalone)
- MLflow model serving / MLServer integration
- Scalable inference backend with load balancing and caching
- Performance optimization

#### Cloud Deployment
- CloudFormation / SageMaker endpoint configuration
- Auto-scaling, monitoring, and cost optimization
- Remote MLflow tracking server with S3 artifact storage

## Technical Specifications

### Model Architecture
```
Input: Manuscript Image (PNG/JPEG)
  ↓
DeepSeek-OCR Vision Encoder
  ↓
Language Model Decoder (with LoRA adapters)
  ↓
Output: Arabic Text
```

### Training Pipeline
```
mssqpi/Arabic-OCR-Dataset (2.16M samples)
  ↓
Conversation Format (User: <image> + prompt, Assistant: text)
  ↓
DeepSeekOCRDataCollator (image preprocessing + tokenization)
  ↓
LoRA Fine-tuning via Unsloth (2% of parameters)
  ↓
Push LoRA adapters to HuggingFace Hub
```

### Inference Pipeline
```
Upload Image → Load Base Model + LoRA Adapters → model.infer() → Arabic Text
```

### Evaluation Pipeline
```
OCR Output → Character/Word Error Rate
           → BLEU Score
           → Diacritic Accuracy
           → Islamic Term Recognition
```

## Arabic-Specific Considerations

- **Right-to-Left text direction**
- **Connected letterforms with contextual shapes**
- **Diacritics preservation for classical texts**
- **Islamic terminology and abbreviations**
- **Historical spelling variations**
- **Multi-column manuscript layouts**

## Success Metrics

### Model Performance
- **Character Error Rate < 5%** for printed text
- **Word Error Rate < 10%** for classical manuscripts
- **Diacritic Accuracy > 90%** for vowelized text
- **Processing Speed < 2 seconds** per page
- **Model Size < 1GB** for deployment efficiency

### MLOps Automation
- **End-to-end automation**: Code push → Auto train → Auto deploy < 1 hour
- **Training cost efficiency**: < $10 per training run on L4 GPU
- **Deployment reliability**: 99.9% uptime with auto-scaling
- **Model versioning**: 100% reproducible experiments
- **Monitoring coverage**: Real-time alerts for performance degradation

## Complete MLOps Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Developer     │    │  GitHub Actions  │    │  HF Spaces GPU  │
│   Push Code     │───▶│  Trigger Train   │───▶│   LoRA Finetune │
│   Update Data   │    │  Run Tests       │    │   MLflow Track  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Production    │◄───│  Model Registry  │◄───│  Auto Evaluate │
│   Deployment    │    │  A/B Testing     │    │  Performance    │
│   Auto-scale    │    │  Version Control │    │  Gate Release   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Current Status

### Completed
- **Phase 1**: Introduction & Setup (Arabic text processing, MLflow integration)
- **Phase 2**: Training Pipeline Development (DeepSeek-OCR + LoRA fine-tuning)
- **Phase 3**: MLOps Automation
  - ✅ GitHub Actions workflow for automated training triggers
  - ✅ HF Spaces training + inference environment with Gradio UI + REST API (L4 GPU)
  - ✅ DeepSeekOCRDataCollator ported from notebook
  - ✅ Auto-push trained LoRA to HF Hub
  - ✅ MLflow experiment tracking (local SQLite on Space)
  - ✅ Inference tab with RTL output + `/api/infer` endpoint
  - ✅ Model loading: base DeepSeek-OCR + LoRA adapters from Hub
  - ✅ Repo consolidation: single repo syncs to HF Spaces via GitHub Actions
  - ✅ `sync-to-hf-spaces.yml` workflow for auto-deploy on push to main

### Up Next
- **Phase 4**: Evaluation & Monitoring Pipeline

---

*Matn - Arabic OCR for classical Islamic texts, powered by DeepSeek-OCR with LoRA fine-tuning.*
