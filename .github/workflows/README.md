# Arabic OCR Training Pipeline - GitHub Actions

This directory contains GitHub Actions workflows for automated Arabic OCR model training.

## Workflow: `arabic-ocr-training.yml`

Automated pipeline that validates code, triggers GPU training on HuggingFace Spaces, and monitors progress.

### 🚀 **Trigger Options**

#### 1. **Automatic Triggers**
- **Push to training files**: Automatically triggers when you push changes to:
  - `pipelines/**` (training code changes)
  - `data/**` (dataset updates)
  - `.github/workflows/arabic-ocr-training.yml` (workflow changes)

#### 2. **Manual Triggers**
Run manually via GitHub Actions UI with custom parameters:

```bash
# Via GitHub CLI
gh workflow run arabic-ocr-training.yml \
  --field num_samples=2000 \
  --field max_steps=100 \
  --field deploy_threshold=0.04
```

### ⚙️ **Configuration Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | `1000` | Number of training samples from dataset |
| `max_steps` | `60` | Training steps (60 ≈ 10 minutes) |
| `model_name` | `unsloth/DeepSeek-OCR` | Base model to fine-tune |
| `experiment_name` | `auto-arabic-ocr` | MLflow experiment name |
| `deploy_threshold` | `0.05` | CER threshold for auto-deployment (5%) |

### 🔄 **Workflow Steps**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  VALIDATE   │───▶│    TRAIN    │───▶│   MONITOR   │───▶│   NOTIFY    │
│             │    │             │    │             │    │             │
│ • Run tests │    │ • Trigger   │    │ • Check     │    │ • Success/  │
│ • Check     │    │   HF Spaces │    │   progress  │    │   Failure   │
│   pipeline  │    │ • Create    │    │ • Wait for  │    │   alerts    │
│ • Validate  │    │   run record│    │   completion│    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 📝 **Required Secrets** (for production)

Add these to your GitHub repository secrets:

```bash
# HuggingFace Token (for Spaces API access)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# HuggingFace Spaces URL (your training space)
HF_SPACE_URL=https://huggingface.co/spaces/your-org/arabic-ocr-trainer
```

### 🎯 **Example Usage Scenarios**

#### **Quick Test Run**
```bash
gh workflow run arabic-ocr-training.yml \
  --field num_samples=100 \
  --field max_steps=10
```

#### **Production Training**
```bash
gh workflow run arabic-ocr-training.yml \
  --field num_samples=5000 \
  --field max_steps=200 \
  --field deploy_threshold=0.03
```

#### **Experiment with Different Model**
```bash
gh workflow run arabic-ocr-training.yml \
  --field model_name=microsoft/trocr-base-arabic \
  --field experiment_name=trocr-experiment
```

### 📊 **Output Artifacts**

The workflow creates:
- **Training run records** in `runs/` directory
- **Configuration files** for reproducibility
- **Links to HuggingFace Spaces** for monitoring
- **GitHub Actions logs** with detailed progress

### 🔍 **Monitoring Training**

1. **GitHub Actions**: Check workflow progress in Actions tab
2. **HuggingFace Spaces**: Visit generated space URL for real-time training logs
3. **MLflow**: Training metrics tracked automatically
4. **Run Records**: JSON files in `runs/` directory with full configuration

### ⚡ **Cost Optimization**

- **Validation runs locally** (free) before triggering GPU training
- **L4 GPU training** (~$0.60/hour, 10 minutes ≈ $0.10 per run)
- **Automatic termination** after training completion
- **Threshold gating** prevents poor models from deploying

### 🛠 **Development Workflow**

1. **Make changes** to training pipeline
2. **Push to GitHub** → Auto-triggers validation
3. **If validation passes** → Training triggered on HF Spaces
4. **Monitor progress** via provided links
5. **Model automatically deploys** if CER < threshold

### 🔧 **Troubleshooting**

- **Validation fails**: Check `test_simplified_pipeline.py` output
- **Training doesn't trigger**: Verify HF_TOKEN and HF_SPACE_URL secrets
- **Training fails**: Check HuggingFace Spaces logs
- **Model doesn't deploy**: Check CER threshold vs actual performance

---

**Next**: Set up HuggingFace Spaces training environment (Phase 3.2)