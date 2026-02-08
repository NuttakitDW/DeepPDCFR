#!/bin/bash
# EC2 User Data script — runs automatically on first boot
# AMI: Deep Learning Base AMI with Single CUDA (Amazon Linux 2023)
# Instance: g4dn.xlarge
# Storage: 50 GB gp3 (breakdown: OS ~10GB, PyTorch CUDA ~5GB, TF ~1.5GB,
#          pip packages ~0.5GB, matrix data ~170MB, repo ~200MB, models ~50MB
#          total ~20GB used, 30GB headroom)
set -euo pipefail

LOG="/var/log/user-data.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== User Data script started at $(date) ==="

# ---- System packages ----
dnf install -y python3.11 python3.11-pip python3.11-devel git tmux

# ---- Clone repo ----
REPO_DIR="/home/ec2-user/DeepPDCFR"
git clone https://github.com/NuttakitDW/DeepPDCFR.git "$REPO_DIR"
cd "$REPO_DIR"
git checkout feat/poker-game

# ---- Unzip data files (required by card_tools.py at import time) ----
cd "$REPO_DIR/matrix"
unzip -o data.zip
cd "$REPO_DIR"

# ---- Create models directory ----
mkdir -p "$REPO_DIR/models/NLHEGeneralized"

# ---- Python venv + dependencies ----
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# ---- Fix ownership (user data runs as root) ----
chown -R ec2-user:ec2-user "$REPO_DIR"

# ---- Verify GPU ----
source venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || echo "GPU check failed"
python -c "from deeppdcfr.card_tools import card_tools; print('card_tools OK, lookup table loaded')" || echo "card_tools check failed"

echo "=== User Data script finished at $(date) ==="
echo "To start training:"
echo "  sudo su - ec2-user"
echo "  cd ~/DeepPDCFR && source venv/bin/activate"
echo "  tmux new -s train"
echo "  python scripts/run.py with configs/NLHEGeneralized.yaml"
