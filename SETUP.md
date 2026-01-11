# Arabic OCR MLOps Setup Guide

Complete setup instructions to get your automated Arabic OCR training pipeline running.

## 🚀 Quick Start Commands

### 1. Create GitHub Repository

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial Arabic OCR pipeline"

# Create GitHub repo (replace YOUR-USERNAME)
gh repo create YOUR-USERNAME/arabic-ocr-mlops --public --source .
git remote add origin https://github.com/YOUR-USERNAME/arabic-ocr-mlops.git
git push -u origin main
```

### 2. Create HuggingFace Space

```bash
# Install HuggingFace CLI
pip install --upgrade huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Create a new Space (replace YOUR-USERNAME)
huggingface-cli repo create YOUR-USERNAME/arabic-ocr-trainer \
  --type space \
  --space_sdk gradio

# Clone the space locally
git clone https://huggingface.co/spaces/YOUR-USERNAME/arabic-ocr-trainer hf-space

# Copy space files
cp -r spaces/arabic-ocr-trainer/* hf-space/
cd hf-space

# Upload to HuggingFace Spaces
git add .
git commit -m "Initial Arabic OCR training space"
git push
```

### 3. Configure GitHub Secrets

In your GitHub repository settings → Secrets and variables → Actions:

```bash
# HuggingFace Token (get from https://huggingface.co/settings/tokens)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Your Space URL
HF_SPACE_URL=https://huggingface.co/spaces/YOUR-USERNAME/arabic-ocr-trainer
```

## 📋 Prerequisites

### Required Accounts
- [x] **GitHub account** - For code repository and CI/CD
- [x] **HuggingFace account** - For GPU training space and model storage

### Required Tools
```bash
# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list
sudo apt update
sudo apt install gh

# Login to GitHub
gh auth login

# Install HuggingFace CLI
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 🏗️ Detailed Setup Steps

### Step 1: Repository Setup

```bash
# Clone this repository or create from scratch
cd /path/to/ml.school

# Initialize git if not already done
git init

# Add all files
git add .
git commit -m "feat: Arabic OCR MLOps pipeline with automated training"

# Create GitHub repository
gh repo create YOUR-USERNAME/arabic-ocr-mlops \
  --description "Automated Arabic OCR training pipeline with MLOps" \
  --public \
  --source .

# Push to GitHub
git remote add origin https://github.com/YOUR-USERNAME/arabic-ocr-mlops.git
git push -u origin main
```

### Step 2: HuggingFace Space Setup

```bash
# Create space on HuggingFace
huggingface-cli repo create YOUR-USERNAME/arabic-ocr-trainer \
  --type space \
  --space_sdk gradio

# Clone space repository
git clone https://huggingface.co/spaces/YOUR-USERNAME/arabic-ocr-trainer hf-space-repo
cd hf-space-repo

# Copy space configuration files
cp -r ../spaces/arabic-ocr-trainer/* .

# Copy main pipeline files (needed for training)
mkdir -p pipelines
cp ../pipelines/arabic_ocr_training_pipeline.py pipelines/
cp ../mlflow_arabic_ocr_config.py .

# Commit and push to trigger space build
git add .
git commit -m "Initial Arabic OCR training space setup"
git push

# Wait for space to build (5-10 minutes)
echo "🏗️  Space building at: https://huggingface.co/spaces/YOUR-USERNAME/arabic-ocr-trainer"
```

### Step 3: Configure GitHub Secrets

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Add repository secrets:

```bash
# Name: HF_TOKEN
# Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxx (from https://huggingface.co/settings/tokens)

# Name: HF_SPACE_URL
# Value: https://huggingface.co/spaces/YOUR-USERNAME/arabic-ocr-trainer
```

### Step 4: Test the Pipeline

```bash
# Test manual trigger
gh workflow run arabic-ocr-training.yml \
  --field num_samples=100 \
  --field max_steps=10 \
  --field experiment_name="test-run"

# Monitor progress
gh run watch

# Or check online
echo "View progress at: https://github.com/YOUR-USERNAME/arabic-ocr-mlops/actions"
```

## 🎯 Usage Examples

### Quick Test (2 minutes, ~$0.02)
```bash
gh workflow run arabic-ocr-training.yml \
  --field num_samples=50 \
  --field max_steps=5
```

### Development Training (10 minutes, ~$0.10)
```bash
gh workflow run arabic-ocr-training.yml \
  --field num_samples=1000 \
  --field max_steps=60
```

### Production Training (30 minutes, ~$0.30)
```bash
gh workflow run arabic-ocr-training.yml \
  --field num_samples=5000 \
  --field max_steps=200 \
  --field deploy_threshold=0.03
```

### Automatic Trigger
```bash
# Just push changes to trigger automatically
git add pipelines/
git commit -m "update: improve training configuration"
git push  # This will automatically trigger training!
```

## 📊 Monitoring & Results

### Training Progress
- **GitHub Actions**: Real-time workflow progress
- **HuggingFace Space**: Training logs and metrics
- **MLflow**: Experiment tracking and model comparison

### Access Points
```bash
# GitHub Actions
https://github.com/YOUR-USERNAME/arabic-ocr-mlops/actions

# HuggingFace Space
https://huggingface.co/spaces/YOUR-USERNAME/arabic-ocr-trainer

# Trained Models
https://huggingface.co/YOUR-USERNAME (auto-uploaded models)
```

## 🔧 Troubleshooting

### Common Issues

**1. Space build fails**
```bash
# Check space logs at your space URL
# Common fix: Update requirements.txt versions
```

**2. GitHub Actions fails**
```bash
# Check that secrets are set correctly
gh secret list

# Verify space URL is accessible
curl https://huggingface.co/spaces/YOUR-USERNAME/arabic-ocr-trainer
```

**3. Training fails**
```bash
# Check space logs for GPU/memory issues
# Reduce num_samples or max_steps for testing
```

### Getting Help
- **Space logs**: Check your HF Space for detailed error messages
- **GitHub Issues**: Create issues in your repository
- **Community**: HuggingFace Discord or GitHub Discussions

## 💰 Cost Estimates

| Configuration | Duration | L4 GPU Cost | Description |
|---------------|----------|-------------|-------------|
| Quick Test    | 2 min    | ~$0.02      | Validation run |
| Development   | 10 min   | ~$0.10      | Standard training |
| Production    | 30 min   | ~$0.30      | High-quality model |

**Note**: L4 GPU costs ~$0.60/hour. Training automatically stops after completion.

## 🎉 You're Ready!

After setup, your pipeline will:
1. ✅ **Auto-trigger** on code changes
2. ✅ **Validate** locally before GPU training
3. ✅ **Train** efficiently with LoRA fine-tuning
4. ✅ **Deploy** automatically if performance meets threshold
5. ✅ **Track** all experiments with MLflow

Push code and watch the magic happen! 🚀