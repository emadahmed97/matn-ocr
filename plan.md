# Arabic OCR for Classical Islamic Texts - Implementation Plan

This document outlines our plan to adapt the ML School codebase from penguin classification to Arabic OCR for classical Islamic texts using Nougat-small architecture.

## Project Overview

We're building an end-to-end machine learning system that can:
- Extract text from classical Arabic Islamic manuscript images
- Provide structured output with proper formatting
- Handle diacritics and classical Arabic conventions
- Deploy as a production-ready service

**Dataset**: MohamedRashad/arabic-books (8,647 Arabic books, 4.8GB text)
**Model**: Microsoft Nougat-small (fine-tuned for Arabic manuscripts)
**Architecture**: Vision Transformer → Text Generation

## Implementation Phases

### Phase 1: Introduction & Setup
Following `.guide/introduction/` structure:

#### 1.1 Environment Setup
- ✅ Install required dependencies (datasets, transformers)
- ✅ Explore Arabic books dataset structure
- 🔲 Set up Nougat model integration
- 🔲 Configure Arabic text processing pipeline

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
Following `.guide/training-pipeline/` structure:

#### 2.1 Data Loading & Preprocessing
- ✅ Replace penguin dataset with `mssqpi/Arabic-OCR-Dataset`
- ✅ Implement Arabic text normalization (reuse from Section 1.2)
- ✅ Use HuggingFace datasets for simple loading
- ✅ Convert dataset to conversation format for fine-tuning

#### 2.2 Model Architecture Setup
- ✅ Use DeepSeek-OCR instead of Nougat (following notebook approach)
- ✅ Configure LoRA fine-tuning for efficient training
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
- ✅ Diacritic accuracy assessment (integrated from Section 1.2)
- ✅ Islamic terminology recognition accuracy (integrated from Section 1.2)

#### 2.6 Model Registration
- ✅ Register best performing models (integrated in training pipeline)
- ✅ Version control for Arabic OCR models (via MLflow tracking)
- ✅ Model metadata and documentation (automated via pipeline)

### Phase 3: MLOps Automation Pipeline
Building automated training and deployment infrastructure:

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

### Phase 5: Model Serving
Following `.guide/serving-model/` structure:

#### 5.1 Local Deployment
- 🔲 MLflow model serving setup
- 🔲 REST API for OCR endpoints
- 🔲 Arabic text response formatting
- 🔲 Local testing and validation

#### 5.2 Production Serving
- 🔲 MLServer integration
- 🔲 Scalable inference backend
- 🔲 Load balancing and caching
- 🔲 Performance optimization

### Phase 6: AWS Deployment
Following `.guide/aws/` structure:

#### 6.1 Infrastructure Setup
- 🔲 CloudFormation templates for OCR
- 🔲 SageMaker endpoint configuration
- 🔲 S3 storage for manuscripts
- 🔲 Network and security setup

#### 6.2 Model Deployment
- 🔲 SageMaker model deployment
- 🔲 Auto-scaling configuration
- 🔲 Monitoring and logging
- 🔲 Cost optimization

#### 6.3 MLflow Remote Setup
- 🔲 Remote MLflow tracking server
- 🔲 S3 artifact storage
- 🔲 Database backend configuration
- 🔲 Access control and security

## Technical Specifications

### Model Architecture
```
Input: Manuscript Image (PNG/JPEG)
  ↓
Vision Transformer Encoder
  ↓
Arabic Language Model Decoder
  ↓
Output: Structured Arabic Text (Markdown)
```

### Dataset Processing
```
Arabic Books Text Corpus
  ↓
Text Normalization & Cleaning
  ↓
Synthetic Image Generation
  ↓
Image-Text Pairs for Training
```

### Evaluation Pipeline
```
OCR Output → Character/Word Error Rate
           → BLEU Score
           → Diacritic Accuracy
           → Islamic Term Recognition
```

## Key Adaptations from Original Codebase

1. **Data Format**: CSV → Image-Text pairs
2. **Model Type**: Classification → Sequence Generation
3. **Evaluation**: Accuracy → CER/WER/BLEU
4. **Features**: Structured columns → Image pixels
5. **Output**: Class labels → Arabic text sequences

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

### Automation Phases
1. **Phase 1**: GitHub Actions + HF Spaces training automation
2. **Phase 2**: Model registry + automated deployment gates
3. **Phase 3**: A/B testing + continuous monitoring
4. **Phase 4**: Auto-retraining + cost optimization

## Next Steps

### ✅ **Completed Phases**
- **Phase 1**: Introduction & Setup (Arabic text processing, MLflow integration)
- **Phase 2**: Training Pipeline Development (DeepSeek-OCR + LoRA fine-tuning)
- **Phase 3 (partial)**: MLOps Automation
  - ✅ GitHub Actions workflow for automated training triggers
  - ✅ HF Spaces training environment with Gradio UI + REST API (L4 GPU)
  - ✅ Real data collator ported from notebook (DeepSeekOCRDataCollator)
  - ✅ Auto-push trained LoRA to HF Hub (`emadahmed97/matn-ocr-arabic-finetuned`)
  - ✅ MLflow experiment tracking (local SQLite on Space)

### 🚀 **Current Phase: Inference Pipeline (Phase 3.4)**
1. **Add Inference tab** to HF Spaces Gradio UI (upload image → OCR text)
2. **Add `/api/infer` REST endpoint** for programmatic inference
3. **Load LoRA model** from HF Hub (`emadahmed97/matn-ocr-arabic-finetuned`)
4. **Handle RTL text** formatting and display

### 📋 **Implementation Priority**
1. Inference tab in Gradio (image upload → model.infer() → Arabic text output)
2. `/api/infer` REST endpoint
3. Model caching (load once, reuse across requests)
4. RTL text formatting and confidence scoring

### 💡 **Key Advantages Achieved**
- **Simplified approach**: DeepSeek-OCR instead of complex Nougat setup
- **Proven dataset**: 2.16M samples from `mssqpi/Arabic-OCR-Dataset`
- **Efficient training**: LoRA fine-tuning (~2 minutes for 10 steps, ~10 minutes for 60 steps)
- **End-to-end pipeline**: Train → Save → Push to Hub, all from Gradio UI
- **Trained model**: https://huggingface.co/emadahmed97/matn-ocr-arabic-finetuned

---

*This plan follows the ML School methodology while adapting for Arabic OCR challenges and opportunities.*