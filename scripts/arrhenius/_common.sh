# ============================================================
# Arrhenius — shared variables sourced by all job scripts
# ============================================================
# Edit this single file when project IDs, paths, or container
# tags change instead of touching each sbatch script.
# ============================================================

# SUPR allocation — update once NAISS approves your project.
# Slurm account name on Arrhenius typically tracks the SUPR ID
# (confirm with `projinfo` at first login).
export NAISS_PROJECT="${NAISS_PROJECT:-naiss2026-X-XXX}"

# Lustre project storage. The path published by NAISS for
# Arrhenius is the single Lustre tree — `projinfo` will print it.
# Set this once, then every script picks it up.
export PROJECT_STORAGE="${PROJECT_STORAGE:-/cfs/klemming/projects/${NAISS_PROJECT}}"

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

# Partition / QoS names — NAISS has not published these yet for
# Arrhenius. The names below are placeholders that follow the
# pattern used on Tetralith/Berzelius; replace once announced.
export GPU_PARTITION="${GPU_PARTITION:-gpu}"
export GPU_QOS="${GPU_QOS:-normal}"

# Helper: run a command inside the container with GPU and our
# extras venv activated, with the project bound in.
arrhenius_run() {
    apptainer exec --nv \
        --bind "${PROJECT_STORAGE}:${PROJECT_STORAGE}" \
        --bind "${CODE_DIR}:${CODE_DIR}" \
        "${CONTAINER_PATH}" \
        bash -lc "source '${EXTRAS_VENV}/bin/activate'; cd '${CODE_DIR}'; $*"
}
