# BENDR Pre-training on LUNARC COSMOS — Step-by-Step Guide

This guide walks you through everything from first login to having a
pre-trained BENDR model ready for fine-tuning. Follow the steps in order.

---

## Before you start — what you need

- [ ] SUPR account approved (you applied already)
- [ ] COSMOS project allocation active (check at https://supr.naiss.se)
- [ ] **Pocket Pass** app installed on your phone (for two-factor authentication)
- [ ] Password set at https://phenix3.lunarc.lu.se/pss
- [ ] Your EDF files accessible (on a drive you can transfer from)

---

## Step 1: Log in to COSMOS

You have two options. Use **Option A** (HPC Desktop) — it's easier.

### Option A: HPC Desktop (recommended)

1. Open the **ThinLinc Client** on your computer
   - Download from: https://www.cendio.com/thinlinc/download
2. Enter the server address: `cosmos-dt.lunarc.lu.se`
3. Enter your LUNARC username and password
4. It will ask for a one-time password — open **Pocket Pass** on your phone and type the code
5. You'll see a Linux desktop. Right-click on the desktop → **Open Terminal**

### Option B: SSH (terminal only)

```bash
ssh cosmos.lunarc.lu.se -l YOUR_USERNAME
```

It will ask for your password, then a one-time code from Pocket Pass.

> **Tip:** If you get disconnected, ThinLinc will reconnect to your existing
> session. SSH won't — you'd need to use `screen` or `tmux`.

---

## Step 2: Upload your code to COSMOS

On your **local Mac**, open Terminal and run:

```bash
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
  ~/Dropbox/Work/eeg-seizure-shared/ \
  YOUR_USERNAME@cosmos.lunarc.lu.se:~/eeg-seizure-shared/
```

This copies the entire project to your home directory on COSMOS.
It will ask for password + Pocket Pass code.

**How long:** A few minutes (just code, not data).

---

## Step 3: Upload your EDF files to COSMOS

Your EDF data is large (~25,000 hours), so it goes into the **project
storage** (not your home directory — home has a quota).

First, find out your project directory. On COSMOS, run:

```bash
projinfo
```

This shows your project name (something like `lu2026-X-XXX`). Your project
storage path is:

```
/lunarc/nobackup/projects/lu2026-X-XXX/
```

Create a folder for the data:

```bash
mkdir -p /lunarc/nobackup/projects/lu2026-X-XXX/edf_data
```

Now transfer the EDF files. On your **local Mac**:

```bash
rsync -avz --progress /path/to/your/edf/files/ \
  YOUR_USERNAME@cosmos.lunarc.lu.se:/lunarc/nobackup/projects/lu2026-X-XXX/edf_data/
```

**How long:** This depends on your connection speed and total data size.
Could be hours or days for terabytes. You can run it overnight — `rsync`
is resumable, so if it stops you can run the same command again and it
picks up where it left off.

> **Important:** The `/lunarc/nobackup/` path means there is NO backup.
> Keep your original files safe elsewhere.

---

## Step 4: Set up the Python environment (one-time)

On COSMOS (in the terminal), run these commands one by one:

```bash
# Go to your project folder
cd ~/eeg-seizure-shared

# Load Anaconda (Python package manager)
module purge
module load Anaconda3/2024.06-1
source config_conda.sh

# Create a new Python environment called "bendr"
conda create -y -n bendr python=3.11

# Activate it
conda activate bendr

# Install PyTorch with GPU support
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies we need
pip install pyedflib scipy numpy tqdm

# Install our project
pip install -e "."
```

**How long:** 5–10 minutes.

### Verify it worked:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

You should see something like: `PyTorch 2.x.x, CUDA: False`

CUDA shows `False` because the login node has no GPU. That's normal.
It will show `True` when running on a GPU compute node.

---

## Step 5: Edit the job scripts

You need to put your project account name into the scripts. Run:

```bash
projinfo
```

Note your project name (e.g., `lu2026-2-123`). Now edit the scripts:

```bash
cd ~/eeg-seizure-shared/scripts/lunarc

# Edit the test script
nano test_gpu.sh
```

In `nano`:
1. Find the line `#SBATCH -A lu20XX-X-XXX`
2. Replace `lu20XX-X-XXX` with your actual project name
3. Find the line `EDF_DIR="/lunarc/nobackup/projects/YOUR_PROJECT/edf_data"`
4. Replace `YOUR_PROJECT` with your project name
5. Press `Ctrl+O` then `Enter` to save
6. Press `Ctrl+X` to exit

Do the same for `pretrain_short.sh`, `pretrain.sh`, and `resume.sh`:

```bash
nano pretrain_short.sh
# same edits as above, save and exit

nano pretrain.sh
# same edits as above, save and exit

nano resume.sh
# same edits as above, save and exit
```

---

## Step 6: Create the logs and output directories

```bash
mkdir -p ~/eeg-seizure-shared/logs
mkdir -p ~/bendr_output
```

---

## Step 7: Run a test job (30 minutes)

This is a quick sanity check. It runs 2 epochs on a small sample to make
sure everything works before you commit to a multi-day training run.

```bash
cd ~/eeg-seizure-shared
sbatch scripts/lunarc/test_gpu.sh
```

You'll see something like: `Submitted batch job 1234567`

### Check if it's running:

```bash
jobinfo -u $USER
```

This shows your job. Status will be:
- `PD` = pending (waiting for a free GPU node)
- `R` = running
- `CG` = completing

### Watch the output live (once it's running):

```bash
tail -f logs/bendr_test_1234567.out
```

(Replace `1234567` with your actual job ID.)

Press `Ctrl+C` to stop watching.

### What to look for in the output:

1. `PyTorch X.X.X, CUDA: True, Device: NVIDIA A100` — GPU is working
2. `Found XX EDF files` — your data was found
3. `Using 8 channel(s) per file` — channels are being used
4. `Epoch 1/2` and `Epoch 2/2` — training is running
5. A checkpoint file saved — checkpointing works

### If something went wrong:

Check the error log:

```bash
cat logs/bendr_test_1234567.err
```

Common issues:
- `ModuleNotFoundError: No module named 'pyedflib'` → you forgot to
  activate the conda environment. Check that the script has the right
  `conda activate bendr` line.
- `No EDF files found` → the `EDF_DIR` path in the script is wrong.
  Double-check with `ls /lunarc/nobackup/projects/YOUR_PROJECT/edf_data/`
- `CUDA not available` → the module load or conda environment is wrong.

---

## Step 8: Short pre-training run (5 epochs)

Instead of jumping straight into a 7-day training job, we do a short run
first. This uses ALL the data but only trains for 5 epochs (~1-2 days).

Why? Two reasons:
1. You can check that the loss is going down (the model is learning)
2. You get an early model to fine-tune in NED-Net and see if it's useful

No work is wasted — the full run later **resumes from this checkpoint**.

```bash
cd ~/eeg-seizure-shared
sbatch scripts/lunarc/pretrain_short.sh
```

### Monitor it:

```bash
# Check status
jobinfo -u $USER

# Watch output (replace job ID with yours)
tail -f logs/bendr_5ep_XXXXXXX.out

# Check how much of your allocation you've used
projinfo
```

### What to look for:

- The **loss should decrease** each epoch (e.g., 8.5 → 6.2 → 5.1 → ...)
- The **accuracy should increase** (e.g., 0.01 → 0.05 → 0.12 → ...)

If the loss is NOT decreasing after 3 epochs, something is wrong — let me
know and we'll debug.

### When it finishes:

You'll get an email. Download the early model to your Mac (see Step 10)
and try fine-tuning it in NED-Net. If the results look promising, continue
to Step 9.

---

## Step 9: Continue to full training (30 epochs)

Once you're happy with the 5-epoch model, continue training to 30 epochs.
The resume script picks up exactly where the short run stopped:

```bash
cd ~/eeg-seizure-shared
sbatch scripts/lunarc/resume.sh
```

This has a **7-day time limit**. If it doesn't finish all 30 epochs in
7 days, just submit it again — it resumes from the latest checkpoint
every time:

```bash
# If it timed out, just run it again:
sbatch scripts/lunarc/resume.sh
```

**Pro tip:** You can chain jobs so the resume starts automatically after
the first one finishes:

```bash
JOBID=$(sbatch --parsable scripts/lunarc/resume.sh)
sbatch -d afterok:$JOBID scripts/lunarc/resume.sh
```

### Alternative: run all 30 epochs from scratch

If you prefer to skip the short run and go straight to 30 epochs:

```bash
sbatch scripts/lunarc/pretrain.sh
```

This is fine too — it saves checkpoints every 5 epochs, so if the job
is stopped you can always resume.

---

## Step 10: Download the trained model

When training finishes (you'll get an email), the best model is at:

```
~/bendr_output/run1/best_model.pt
```

Download it to your Mac. On your **local Mac**, open Terminal:

```bash
# Create the folder where the app looks for pre-trained weights
mkdir -p ~/.eeg_seizure_analyzer/pretrained

# Download the model
scp YOUR_USERNAME@cosmos.lunarc.lu.se:~/bendr_output/run1/best_model.pt \
  ~/.eeg_seizure_analyzer/pretrained/bendr_rodent_25k.pt

# Also grab the training log (optional, for reference)
scp YOUR_USERNAME@cosmos.lunarc.lu.se:~/bendr_output/run1/pretrain_log.json \
  ~/.eeg_seizure_analyzer/pretrained/
```

---

## Step 11: Use the pre-trained model in NED-Net

Now that you have the pre-trained weights on your Mac:

1. Open NED-Net (`python -m eeg_seizure_analyzer.dash_app.main`)
2. Go to **Training** tab → **Dataset / Model**
3. Select **Architecture: BENDR**
4. In **Pre-trained weights**, you should see `bendr_rodent_25k` in the dropdown
5. Annotate some seizures, build a dataset, and click **Start Training**
6. Once trained, go to **Detection → Seizure** → select **BENDR** → pick your model → **Detect**

---

## Quick reference: useful commands on COSMOS

| What you want to do | Command |
|---------------------|---------|
| Check your jobs | `jobinfo -u $USER` |
| Cancel a job | `scancel JOBID` |
| Check disk space | `snicquota` |
| Check project allocation | `projinfo` |
| Watch job output live | `tail -f logs/bendr_pretrain_JOBID.out` |
| See what's in a directory | `ls -la /path/to/dir` |
| Check job estimated start time | `squeue --start -j JOBID` |
| See all available GPUs | `sinfo -p gpua100` |

---

## Troubleshooting

### "My job is stuck in PD (pending) for hours"

The GPU nodes are shared with other users. Check estimated start:

```bash
squeue --start -j YOUR_JOBID
```

If the wait is very long, it might be because all A100 nodes are busy.
This is normal — the job will start when a node frees up.

### "My job failed immediately"

Check the error log:

```bash
cat logs/bendr_pretrain_JOBID.err
```

### "I ran out of disk space"

```bash
snicquota
```

If your home is full, move large files to project storage:

```bash
mv ~/large_file /lunarc/nobackup/projects/YOUR_PROJECT/
```

### "I forgot my project account name"

```bash
projinfo
```

### "I want to check if my data transferred correctly"

```bash
ls /lunarc/nobackup/projects/YOUR_PROJECT/edf_data/ | head -20
ls /lunarc/nobackup/projects/YOUR_PROJECT/edf_data/ | wc -l
```

The first command shows the first 20 files. The second counts total files.
