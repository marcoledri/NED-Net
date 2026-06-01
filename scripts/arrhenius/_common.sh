# ============================================================
# Arrhenius — shared variables sourced by all job scripts
# ============================================================
# Edit this single file when project IDs, paths, or container
# tags change instead of touching each sbatch script.
# ============================================================

# SUPR allocation — update once NAISS approves your project.
# Slurm account name on Arrhenius matches the SUPR ID. Verify what
# you've been granted at https://supr.naiss.se/account/ and via
# `storagequota` on a login node.
export NAISS_PROJECT="${NAISS_PROJECT:-naiss2026-3-358}"

# Project storage on Arrhenius. NAISS publishes two tiers:
#   /nobackup/proj/disk/<PROJECT>   – default bulk storage (slower, larger)
#   /nobackup/proj/flash/<PROJECT>  – fast scratch (use if granted)
# Run `storagequota` on a login node to see which paths your project
# has and how much quota remains. Override PROJECT_STORAGE if your
# allocation only includes the flash tier (or both, and you prefer it
# for hot data).
export PROJECT_STORAGE="${PROJECT_STORAGE:-/nobackup/proj/disk/${NAISS_PROJECT}}"

# Data, code, output, container.
export EDF_DIR="${EDF_DIR:-${PROJECT_STORAGE}/edf_data}"
export CODE_DIR="${CODE_DIR:-${HOME}/NED-Net}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_STORAGE}/bendr_output/run1}"
export CONTAINER_PATH="${CONTAINER_PATH:-${PROJECT_STORAGE}/containers/pytorch-ngc-arm64.sif}"
export EXTRAS_VENV="${EXTRAS_VENV:-${PROJECT_STORAGE}/venvs/bendr-extras}"

# Local NVMe per node (Arrhenius advertises ~1.8 TB). Use it to
# stage the EDF index / scratch caches so we don't hammer Lustre
# from every worker process.
export NVME_SCRATCH="${NVME_SCRATCH:-/scratch/local/${SLURM_JOB_ID:-tmp}}"

# Partition / QoS. The Arrhenius GPU partition (Grace Hopper / H200)
# is `arrhenius-gpu`. No dedicated short-job QoS is documented yet, so
# we leave QOS unset by default and rely on walltime to gate test jobs.
export GPU_PARTITION="${GPU_PARTITION:-arrhenius-gpu}"
export GPU_QOS="${GPU_QOS:-}"

# Helper: run a command inside the container with GPU and our
# extras venv activated, with the project bound in.
arrhenius_run() {
    apptainer exec --nv \
        --bind "${PROJECT_STORAGE}:${PROJECT_STORAGE}" \
        --bind "${CODE_DIR}:${CODE_DIR}" \
        "${CONTAINER_PATH}" \
        bash -lc "source '${EXTRAS_VENV}/bin/activate'; cd '${CODE_DIR}'; $*"
}
