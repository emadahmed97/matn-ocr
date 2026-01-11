# Arabic OCR for Classical Islamic Texts - Updated Implementation Plan

*Based on existing `notebooks/arabic_ocr_finetune.ipynb` with DeepSeek-OCR + Unsloth*

## Project Overview

We have a working Arabic OCR system using:
- **Model**: DeepSeek-OCR (fine-tuned with Unsloth)
- **Dataset**: `mssqpi/Arabic-OCR-Dataset`
- **Method**: LoRA fine-tuning with custom data collator
- **Performance**: 74% CER improvement (23% → 6% on sample)

## Current State Analysis

### ✅ Already Completed (from notebook):
1. **Model Setup**: DeepSeek-OCR with Unsloth integration
2. **Dataset**: Arabic OCR dataset (2,160,000 samples)
3. **Data Processing**: Custom `DeepSeekOCRDataCollator`
4. **Training Pipeline**: LoRA fine-tuning (77M/3.4B params trained)
5. **Inference**: Working OCR with dynamic image preprocessing
6. **Model Saving**: Both LoRA adapters and merged 16-bit model

### 🔲 Missing for ML School Integration:
1. **Metaflow Pipeline Structure**: Convert notebook → production pipelines
2. **MLflow Integration**: Experiment tracking and model registry
3. **Cross-validation**: Adapt for OCR evaluation metrics
4. **Monitoring Pipeline**: Model quality and data drift detection
5. **Deployment Pipeline**: API serving and scaling
6. **Testing Framework**: Automated evaluation and CI/CD

## Updated Implementation Plan

### Phase 1: Pipeline Extraction & Adaptation
*Convert working notebook code to ML School structure*

#### 1.1 Extract Core Components ✅
- ✅ DeepSeek-OCR model loading
- ✅ Custom data collator
- ✅ Training configuration
- ✅ Inference pipeline
- 🔲 Create reusable modules from notebook cells

#### 1.2 Create Training Pipeline
- 🔲 `pipelines/arabic_training.py` - Replace penguin training
- 🔲 Integrate `DeepSeekOCRDataCollator`
- 🔲 Add MLflow experiment tracking
- 🔲 Implement OCR evaluation metrics (CER, WER, BLEU)
- 🔲 Cross-validation for Arabic OCR

#### 1.3 Dataset Integration
- 🔲 Replace penguins dataset with `mssqpi/Arabic-OCR-Dataset`
- 🔲 Add data preprocessing utilities
- 🔲 Implement image-text pair handling
- 🔲 Create data quality validation

### Phase 2: ML School Pipeline Adaptation

#### 2.1 Training Pipeline (`pipelines/training.py`)
```python
# Key adaptations needed:
- DatasetMixin → ArabicOCRDatasetMixin
- build_model() → load_deepseek_ocr_model()
- Classification metrics → OCR metrics (CER/WER)
- Cross-validation → Text-based splitting
- MLflow logging → OCR-specific artifacts
```

#### 2.2 Inference Pipeline (`pipelines/inference/`)
```python
# Integrate from notebook:
- model.infer() method
- Dynamic image preprocessing
- Custom PyFunc wrapper for MLflow
- Arabic text post-processing
```

#### 2.3 Monitoring Pipeline (`pipelines/monitoring.py`)
```python
# New components:
- OCR accuracy drift detection
- Character/word error rate tracking
- Arabic text quality validation
- Image quality assessment
```

### Phase 3: Production Integration

#### 3.1 Model Registry & Versioning
- 🔲 Register fine-tuned DeepSeek-OCR models
- 🔲 Version LoRA adapters and merged models
- 🔲 Model metadata and performance tracking
- 🔲 A/B testing framework for OCR models

#### 3.2 Serving & Deployment
- 🔲 MLflow serving integration
- 🔲 REST API for OCR endpoints
- 🔲 Batch processing capabilities
- 🔲 Performance optimization (GPU/CPU)

#### 3.3 AWS Deployment
- 🔲 SageMaker endpoint for DeepSeek-OCR
- 🔲 S3 storage for manuscript images
- 🔲 Lambda functions for preprocessing
- 🔲 CloudFormation templates

## Key Technical Components to Extract

### From Notebook Cell #3: Model Loading
```python
from unsloth import FastVisionModel
from transformers import AutoModel

model, tokenizer = FastVisionModel.from_pretrained(
    "./deepseek_ocr",
    load_in_4bit=False,
    auto_model=AutoModel,
    trust_remote_code=True,
    use_gradient_checkpointing="unsloth"
)
```

### From Notebook Cell #22: Data Collator
```python
class DeepSeekOCRDataCollator:
    # Full implementation for image-text processing
    # Dynamic preprocessing with crop modes
    # Attention mask and label creation
```

### From Notebook Cell #24: Training Configuration
```python
trainer = Trainer(
    model=model,
    data_collator=DeepSeekOCRDataCollator(...),
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        # OCR-specific training params
    )
)
```

## Immediate Next Steps

### Step 1: Extract and Modularize
1. **Create `pipelines/arabic_ocr/`** - New module structure
2. **Extract model loading** → `arabic_ocr/model.py`
3. **Extract data collator** → `arabic_ocr/data_collator.py`
4. **Extract preprocessing** → `arabic_ocr/preprocessing.py`

### Step 2: Adapt Existing Pipelines
1. **Modify `pipelines/training.py`**:
   - Replace DatasetMixin with ArabicOCRDatasetMixin
   - Change evaluation metrics to CER/WER
   - Integrate DeepSeek-OCR model loading

2. **Update `pipelines/inference/`**:
   - Replace classification with OCR inference
   - Add image preprocessing pipeline
   - Implement text post-processing

### Step 3: Testing & Validation
1. **Create test suite** based on notebook results
2. **Benchmark against notebook performance** (6% CER target)
3. **Validate MLflow integration**
4. **Test cross-validation strategy**

## Success Criteria

### Performance Targets (from notebook):
- ✅ **Character Error Rate < 6%** (already achieved)
- ✅ **Training Efficiency**: 1.6GB memory for LoRA training
- ✅ **74% CER improvement** over baseline
- 🔲 **End-to-end pipeline** < 2 minutes training time
- 🔲 **Inference speed** < 2 seconds per image

### Integration Targets:
- 🔲 **Seamless MLflow tracking** for OCR experiments
- 🔲 **Production deployment** with auto-scaling
- 🔲 **Monitoring dashboard** for Arabic OCR quality
- 🔲 **CI/CD pipeline** with automated testing

## File Structure Changes

```
pipelines/
├── arabic_ocr/                    # New: Extracted from notebook
│   ├── __init__.py
│   ├── model.py                   # DeepSeek-OCR loading
│   ├── data_collator.py          # Custom OCR data collator
│   ├── preprocessing.py          # Image processing utilities
│   ├── metrics.py                # CER, WER, BLEU evaluation
│   └── inference.py              # OCR inference pipeline
├── training.py                    # Modified: Use Arabic OCR
├── inference/
│   ├── model.py                   # Modified: OCR PyFunc model
│   └── backend.py                 # Modified: Arabic text serving
└── monitoring.py                  # Modified: OCR quality monitoring
```

---

*This plan leverages the proven notebook implementation while adapting it to the ML School production framework.*