#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/user_data.log) 2>&1

# --- System packages ---
yum update -y
yum install -y git htop screen gcc gcc-c++

# --- Python 3.11 via Miniconda (avoids AL2 OpenSSL issues) ---
curl -sLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda
export PATH="/opt/miniconda/bin:$PATH"
echo 'export PATH="/opt/miniconda/bin:$PATH"' >> /home/ec2-user/.bashrc
python --version

# --- Clone repo and checkout branch ---
cd /home/ec2-user
git clone https://github.com/NuttakitDW/DeepPDCFR.git
cd DeepPDCFR
git checkout optimize/paralellization

# --- Install Python dependencies ---
pip install --upgrade pip
pip install -e .

# --- Verify ---
python -c "from deeppdcfr.parallel import run_parallel_dfs; print('IMPORT OK')"

# --- Ready marker ---
echo "SETUP COMPLETE $(date)" > /home/ec2-user/setup_done.txt
chown -R ec2-user:ec2-user /home/ec2-user/DeepPDCFR /home/ec2-user/setup_done.txt /opt/miniconda
