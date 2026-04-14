#!/bin/bash
# ============================================================
# LUNARC COSMOS — Resume BENDR pre-training
# ============================================================
# If the previous job hit the 7-day walltime limit, this script
# resumes from the latest checkpoint.
#
# Usage:
#   sbatch scripts/lunarc/resume.sh
#
# Or chain automatically after the first job:
#   JOBID=$(sbatch --parsable scripts/lunarc/pretrain.sh)
#   sbatch -d afterok:$JOBID scripts/lunarc/resume.sh
#
# BEFORE RUNNING: Edit the lines marked EDIT below.
# ============================================================

#SBATCH -p gpua100
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH -J bendr_resume
#SBATCH -o logs/bendr_resume_%j.out
#SBATCH -e logs/bendr_resume_%j.err
#SBATCH -A lu20XX-X-XXX                   # <-- EDIT: your project account
#SBATCH --mail-user=marco.ledri@med.lu.se
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

echo "========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "========================================="

# Activate environment
module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr

nvidia-smi

cd $HOME/eeg-seizure-shared
mkdir -p logs

# EDIT: must match pretrain.sh
EDF_DIR="/lunarc/nobackup/projects/YOUR_PROJECT/edf_data"
OUTPUT_DIR="$HOME/bendr_output/run1"

# Find the latest checkpoint automatically
LATEST_CKPT=$(ls -t "$OUTPUT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $OUTPUT_DIR"
    echo "Run pretrain.sh first."
    exit 1
fi

echo "Resuming from: $LATEST_CKPT"

python -m eeg_seizure_analyzer.ml.bendr_pretrain \
    --data-dir "$EDF_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --channels 0 1 2 3 4 5 6 7 \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --num-workers 16 \
    --segment-sec 60 \
    --target-fs 256 \
    --encoder-h 512 \
    --context-layers 8 \
    --context-heads 8 \
    --checkpoint-every 5 \
    --val-fraction 0.05 \
    --resume "$LATEST_CKPT"

echo "========================================="
echo "Resume finished at $(date)"
echo "========================================="
