#!/bin/bash
#SBATCH --job-name=Clean_EVADIAB
#SBATCH --output=logs/Clean_EVADIAB_log_%j.log
#SBATCH --error=logs/Clean_EVADIAB_log_%j.log
#SBATCH --cpus-per-task=4

# Load the correct CUDA module
module load /softs/modules/cuda/12.8.1_570.124.06

# Activate the correct environment
source ~/phd_v1_env/bin/activate

# Navigate to the script directory
cd ~/EVADIAB/

# ---- W&B storage location ----
export WANDB_DIR="$(pwd)/outputs"
mkdir -p outputs

# Confirm correct Python environment
echo "✅ Current Python environment: $(which python)"

# Run dataset verification script
python -m src.preprocess.clean \
  --input data/raw/EVADIAB_clinical_Data.xlsx \
  --output data/processed/evadiab_clinical_clean.csv \
  --sheet 0
