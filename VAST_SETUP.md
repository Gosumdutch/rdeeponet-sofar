# Vast.ai Setup Guide

## 1. Local: Push code to Git

```bash
# Add remote (GitHub/GitLab)
git remote add origin https://github.com/YOUR_USERNAME/rdeeponet-sofar.git

# Commit and push
git add -A
git commit -m "Physics-informed loss implementation"
git push -u origin master
```

## 2. Google Drive: Upload data

Upload `data_h5.zip` (1.2GB) to Google Drive and get shareable link.

## 3. Vast.ai: Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/rdeeponet-sofar.git
cd rdeeponet-sofar

# Create conda env
conda create -n wcsmo python=3.10 -y
conda activate wcsmo

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download data from Google Drive
pip install gdown
gdown --fuzzy "YOUR_GOOGLE_DRIVE_LINK" -O data_h5.zip

# Extract data
mkdir -p R-DeepONet_Data/data
unzip data_h5.zip -d R-DeepONet_Data/data/

# Verify
ls R-DeepONet_Data/data/h5/ | wc -l  # Should be 4965
```

## 4. Run Training

```bash
# Quick test (5 epochs)
python run_comparison.py --config config_train.yaml --quick

# Full training - Value only
python train_runner.py --config config_value_only.yaml

# Full training - Physics-informed
python train_runner.py --config config_stage2_highres.yaml

# Comparison experiment
python run_comparison.py --config config_stage2_highres.yaml --output_dir experiments/comparison
```

## 5. Cursor SSH Connection

1. Vast.ai instance → "Connect" → Copy SSH command
2. Cursor → Remote Explorer → Add SSH Host
3. Open remote folder: `/root/rdeeponet-sofar`

## Expected Training Time (RTX 4090)

| Config | Epochs | Batch | Est. Time |
|--------|--------|-------|-----------|
| Quick test | 5 | 8 | ~3 min |
| Value-only | 60 | 16 | ~25 min |
| Physics | 60 | 16 | ~30 min |
| Comparison | 60×2 | 16 | ~1 hour |

