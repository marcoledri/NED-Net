# ============================================================
# Local CPU smoke test for BENDR pre-training (Windows / PowerShell)
# ============================================================
# Runs the same bendr_pretrain command that LUNARC's
# test_gpu_tiny.sh runs, but on your laptop's CPU. Catches
# import / path / argparse / EDF-reader bugs in minutes,
# without waiting in the COSMOS test queue.
#
# Usage:
#   .\scripts\local\test_tiny.ps1 -EdfDir C:\path\to\your\edfs
#
# Optional:
#   -Epochs 2
#   -OutputDir .\bendr_test_output
# ============================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EdfDir,

    [int]$Epochs = 2,

    [string]$OutputDir = (Join-Path (Get-Location) "bendr_test_output")
)

$ErrorActionPreference = "Stop"

# Locate repo root (this script lives at scripts/local/)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

# Activate the project venv. The user installs into .venv with
# Python 3.12 (see memory: pyedflib has no 3.13/3.14 Windows wheels).
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Could not find .venv at $VenvActivate. Create it with 'py -3.12 -m venv .venv' then 'pip install -e .'."
}
. $VenvActivate

# Sanity-check inputs before launching the trainer
if (-not (Test-Path $EdfDir)) {
    Write-Error "EDF directory not found: $EdfDir"
}

$edfCount = (Get-ChildItem -Path $EdfDir -Filter *.edf -File -ErrorAction SilentlyContinue).Count
if ($edfCount -lt 1) {
    Write-Error "No .edf files in $EdfDir. Convert some adicht files first via the NED-Net Tools tab."
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Local BENDR smoke test" -ForegroundColor Cyan
Write-Host "  EDF dir   : $EdfDir ($edfCount file(s))"
Write-Host "  Output    : $OutputDir"
Write-Host "  Epochs    : $Epochs"
Write-Host "  Device    : CPU (no CUDA on this run)"
Write-Host "=========================================" -ForegroundColor Cyan

# Pin to CPU so we don't accidentally try to use an iGPU.
$env:CUDA_VISIBLE_DEVICES = ""

# Same flags as test_gpu_tiny.sh (workers=2, val_fraction=0,
# segments_per_file=3, batch=16) plus CPU-realistic batch / workers.
python -m eeg_seizure_analyzer.ml.bendr_pretrain `
    --data-dir $EdfDir `
    --output-dir $OutputDir `
    --channels 0 1 2 3 4 5 6 7 `
    --epochs $Epochs `
    --batch-size 8 `
    --num-workers 0 `
    --segments-per-file 3 `
    --val-fraction 0 `
    --checkpoint-every 1

if ($LASTEXITCODE -ne 0) {
    Write-Error "bendr_pretrain exited with code $LASTEXITCODE -- see output above."
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Done. Checkpoint(s) in: $OutputDir" -ForegroundColor Green
Write-Host "Loss should have decreased between the two epochs." -ForegroundColor Green
Write-Host "If yes, the LUNARC run is the same command on a GPU." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
