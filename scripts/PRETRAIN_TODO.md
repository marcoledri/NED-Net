# Pre-training Notes

## Resolved: `--target-fs` aligned to 250 across pre-training scripts

The pre-training sample rate now matches fine-tuning / inference at **250 Hz**
(was 256 Hz). The Fourier `resample` in `bendr_pretrain.py` was also swapped
for integer-factor `decimate` (zero-phase, anti-aliased) to mirror the
`_downsample` path in `dataset.py` / `predict.py`. With native 2 kHz EDFs the
decimation factor is exactly 8.

Scripts updated: `scripts/{lunarc,arrhenius}/{pretrain,pretrain_short,resume}.sh`.

## Background (why the model still works on 2 kHz data)

- The model never sees 2 kHz — `predict.py` downsamples new recordings automatically
  before detection, so native rate is irrelevant as long as inference matches training.
- Downsampling to 250 Hz keeps content below ~125 Hz. Standard seizure morphology
  (spike-wave, rhythmic spiking) is preserved; HFOs > 125 Hz are discarded (not used
  by this model).

## Allocation sanity check (Arrhenius)

- 400 GPU-h/month + 1.7 TB is enough for one full campaign (test + 5-epoch short +
  30-epoch full ≈ 150–280 GPU-h on a single A100; ~4–8 h/epoch).
- Dataset is ~25,000 **channel-hours** total — storage is ample even at native 2 kHz.
- Confirm jobs are billed **per-GPU, not per-node** (request 1 GPU explicitly), or the
  quota burns several× faster. Expect one resume (30 epochs may exceed walltime) —
  handled by checkpoint-based resume.
