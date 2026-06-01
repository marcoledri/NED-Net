#!/bin/bash
# ============================================================
# Arrhenius (NAISS) — One-time environment setup for BENDR
# ============================================================
# Arrhenius's GPU partition uses NVIDIA GH200 superchips: ARM
# (aarch64) CPU fused with H100 GPU, 96 GB HBM. Standard x86
# conda recipes don't translate cleanly. The robust path is the
# NVIDIA NGC PyTorch container via Apptainer — it ships the
# vendor-tuned PyTorch + CUDA + NCCL build for GH200.
#
# This script pulls the container ONCE and stores it under your
# project storage. Job scripts mount it read-only.
#
# Usage (on an Arrhenius login node):
#   cd ~/NED-Net
#   bash scripts/arrhenius/setup_env.sh
#
# TODO before first run:
#   - Set NAISS_PROJECT below to the SUPR project ID (e.g. naiss2026-X-XXX)
#   - If `storagequota` reports a path other than /nobackup/proj/disk/<id>,
#     override PROJECT_STORAGE below (or via the env var) — typically only
#     needed if you were granted the flash tier instead of disk
#   - Confirm Apptainer is available as a module (it's standard at NAISS)
# ============================================================

set -euo pipefail

# ── EDIT: SUPR project for Arrhenius once the allocation is approved ──
NAISS_PROJECT="${NAISS_PROJECT:-naiss2026-X-XXX}"

# Arrhenius project storage convention (per NAISS docs):
#   /nobackup/proj/disk/<PROJECT>   – default bulk
#   /nobackup/proj/flash/<PROJECT>  – fast scratch (use if granted)
# Run `storagequota` on a login node to see which directories your
# project actually has.
PROJECT_STORAGE="${PROJECT_STORAGE:-/nobackup/proj/disk/${NAISS_PROJECT}}"

CONTAINER_DIR="${PROJECT_STORAGE}/containers"
CONTAINER_PATH="${CONTAINER_DIR}/pytorch-ngc-arm64.sif"

# NGC PyTorch tag — pick a known-good monthly release. 24.10 is the
# first tag that ships PyTorch 2.5 + CUDA 12.6 for sbsa/aarch64 with
# GH200-specific NCCL tunings; newer tags are fine but pin one for
# reproducibility. Update intentionally, not silently.
NGC_TAG="${NGC_TAG:-24.10-py3}"
NGC_URI="docker://nvcr.io/nvidia/pytorch:${NGC_TAG}"

echo "=== Arrhenius BENDR setup ==="
echo "Project storage : ${PROJECT_STORAGE}"
echo "Container path  : ${CONTAINER_PATH}"
echo "NGC image       : ${NGC_URI}"
echo

if [ ! -d "${PROJECT_STORAGE}" ]; then
    echo "ERROR: project storage not found at ${PROJECT_STORAGE}"
    echo "Run \`storagequota\` and set PROJECT_STORAGE to the path it reports,"
    echo "then re-run this script."
    exit 1
fi

mkdir -p "${CONTAINER_DIR}"

# Load Apptainer if it's available as a module. Some NAISS sites also
# expose it directly on PATH.
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load Apptainer 2>/dev/null || \
        echo "Note: no Apptainer module found, expecting it on PATH"
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not on PATH. Check the module name (some sites"
    echo "       use 'singularity' instead of 'apptainer')."
    exit 1
fi

if [ -f "${CONTAINER_PATH}" ]; then
    echo "Container already present — skipping pull."
    echo "Delete ${CONTAINER_PATH} and re-run if you need to refresh."
else
    echo "Pulling NGC PyTorch (aarch64). This may take 10–30 min."
    apptainer pull "${CONTAINER_PATH}" "${NGC_URI}"
fi

# Install our project's pure-Python deps into a venv that sits OUTSIDE
# the container and gets bind-mounted in at run time. We do this so we
# can update pyedflib / our own package without rebuilding the SIF.
VENV_PATH="${PROJECT_STORAGE}/venvs/bendr-extras"
echo
echo "Creating extras venv inside the container at: ${VENV_PATH}"

apptainer exec --nv "${CONTAINER_PATH}" bash -lc "
    set -euo pipefail
    python -m venv --system-site-packages '${VENV_PATH}'
    source '${VENV_PATH}/bin/activate'
    pip install --no-cache-dir --upgrade pip
    pip install --no-cache-dir pyedflib scipy tqdm
    cd ${HOME}/NED-Net
    pip install --no-cache-dir -e .
"

echo
echo "=== Setup complete ==="
echo
echo "Quick smoke test (single-node, 1 GPU):"
echo "  apptainer exec --nv ${CONTAINER_PATH} \\"
echo "      bash -lc 'source ${VENV_PATH}/bin/activate; \\"
echo "                python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"'"
echo
echo "Then submit:  sbatch scripts/arrhenius/test_gpu.sh"
