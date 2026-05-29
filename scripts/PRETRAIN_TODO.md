# Pre-training TODO / Notes

## ⚠️ FIX before running BENDR pre-training on Arrhenius: align `--target-fs` to 250

**Do this on the Windows machine, in the Arrhenius job scripts** (they live there and
were never pushed — the repo only has `scripts/lunarc/*`).

### The problem

The sample rate is inconsistent across the pipeline:

| Stage | File | Rate | Method |
|-------|------|------|--------|
| BENDR pre-training | `eeg_seizure_analyzer/ml/bendr_pretrain.py:152` | **256 Hz** | `scipy.signal.resample` (Fourier) |
| Fine-tuning dataset | `eeg_seizure_analyzer/ml/dataset.py:33` (`TARGET_FS = 250`) | **250 Hz** | `scipy.signal.decimate` (zero-phase) |
| Inference | `eeg_seizure_analyzer/ml/predict.py:70` | **250 Hz** | `decimate` |

The pre-trained encoder learns features at 256 Hz but is then fine-tuned and run at
250 Hz — a ~2.3% time/frequency scale shift, for no benefit.

### The fix

In the Arrhenius pre-training scripts (and the LUNARC ones below), change:

```
--target-fs 256   →   --target-fs 250
```

Why 250: source EDFs are native **2 kHz**, and 2000 / 250 = **8** (clean integer
decimation). 2000 / 256 = 7.8125 (non-integer, forces Fourier resample).

Optional secondary cleanup: make `bendr_pretrain.py` use the same `decimate` path as
`dataset.py` / `predict.py` so the resampling *method* also matches. The rate match is
the part that matters most.

LUNARC scripts in this repo that carry the same `--target-fs 256` and should be fixed
when re-syncing: `scripts/lunarc/pretrain.sh`, `pretrain_short.sh`, `resume.sh`.

### Background (why the model still works on 2 kHz data)

- The model never sees 2 kHz — `predict.py` downsamples new recordings automatically
  before detection, so native rate is irrelevant as long as inference matches training.
- Downsampling to 250 Hz keeps content below ~125 Hz. Standard seizure morphology
  (spike-wave, rhythmic spiking) is preserved; HFOs > 125 Hz are discarded (not used
  by this model).

### Allocation sanity check (Arrhenius)

- 400 GPU-h/month + 1.7 TB is enough for one full campaign (test + 5-epoch short +
  30-epoch full ≈ 150–280 GPU-h on a single A100; ~4–8 h/epoch).
- Dataset is ~25,000 **channel-hours** total — storage is ample even at native 2 kHz.
- Confirm jobs are billed **per-GPU, not per-node** (request 1 GPU explicitly), or the
  quota burns several× faster. Expect one resume (30 epochs may exceed walltime) —
  handled by checkpoint-based resume.
