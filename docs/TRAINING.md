# Water segmentation: training and inference

Step 11 implements the model, losses, training artifacts and inference. No real
training or performance measurement was run while implementing it. All model
tests use synthetic data and `encoder_weights=None`. We do not yet know which
experiment below wins. Outputs estimate **water**, including permanent water;
a post-only model does not establish new flooding, damage or road failure.

## Data and the limits of the reported scores

Use the local Sen1Floods11 `S1Hand/`, `LabelHand/` and
`splits/flood_handlabeled/` directories. The index uses the official CSVs without
random splitting: train=252, val=89, test=90, and Bolivia=15 chips. Train, val and
test share the same ten regions. **Only Bolivia is a region holdout**, so it is
the closest generalization number, but 15 chips is small. Test and Bolivia are
always reported separately. This caveat is also written inside `metrics.json`.

SAR inputs are named VV/VH dB bands, clipped to [-35, 5], then normalized by fixed
full-training-chip means/stds. Missing `stats.json` is regenerated from TRAIN
only; validation and heldout pixels never contribute. Stats are copied into
every checkpoint; inference never reads a dataset's current stats file. Labels
-1 and declared nodata become ignore index 255, including one local mask that
declares nodata=0. Nonfinite SAR pixels are also ignored. All-ignore batches skip
the optimizer entirely. An empty validation mask cannot yield a successful model
selection score. ECE measures water-probability calibration on labeled pixels;
it does not turn probabilities into operational certainty.

## Local setup and smoke run

Use your existing venv. If torch/SMP imports fail, the user-run install command is:

```sh
.venv/bin/python -m pip install torch torchvision segmentation-models-pytorch
```

If a compatible wheel is unavailable for Python 3.14, use a separate 3.13 venv
(assuming `python3.13` is installed):

```sh
python3.13 -m venv .venv313
.venv313/bin/python -m pip install -r requirements.txt
```

Substitute `.venv313/bin/python` in subsequent commands. Torch is required;
an SMP import failure is reported and `--backend auto` uses a compact pure-torch
U-Net instead. Its encoder is not ResNet34 and it cannot use ImageNet weights.
`--backend smp` requires SMP and fails clearly rather than changing backend.
`--backend fallback --depth 3 --width 16` explicitly selects the compact model.
The backend and dimensions are stored in the checkpoint, so moving machines
cannot silently rebuild a different network. Loading any checkpoint requests
no pretrained weights and does not download anything.

```sh
.venv/bin/python -m src.model.train --data-root data/external/sen1floods11 --out-dir runs/smoke --smoke --no-pretrained
```

Smoke uses two epochs, batch size 1, 64-pixel crops, two batches per split, and no
pretrained weights. Aim for roughly a minute on a Mac; hardware/import/cache
costs vary and runtime has not been measured on real data. **Smoke metrics are
limited-batch diagnostics**, labeled as such, not full-split results. Training
stats and class balance still use full TRAIN chips. Use a fresh output folder
when rerunning, or resume explicitly.

Devices auto-select CUDA, then MPS, then CPU; `--device cpu` forces CPU. Only CUDA
uses AMP. MPS fallback is set before the training entry point imports torch.
Unsupported operations may use CPU; computations sent to MPS use float32, never
float64. NumPy's statistics and metric accumulators stay on CPU. Worker count
defaults to zero to avoid macOS spawn issues. A first full local run can use
batch size 2 and 256-pixel crops on a 16 GB Mac:

```sh
.venv/bin/python -m src.model.train --data-root data/external/sen1floods11 --out-dir runs/local_a --epochs 30 --batch-size 2 --crop-size 256 --encoder resnet34 --pretrained --no-augment --seed 0
```

ImageNet weights may download **when you explicitly run `--pretrained`**. The
default and `--no-pretrained` do not request them. A backend import error is a
fallback trigger; download failures and invalid encoder configurations are
errors, not reasons to pretend a pretrained run succeeded.

## First experiments

Keep splits, seed, optimizer, channels and budgets identical. These first
experiments disable geometric augmentation to isolate the named differences.
`--no-augment` disables flips/rotations; radiometric flags remain independently
controlled and default to zero.

| Experiment | Encoder initialization | Radiometry | Additional flags |
| --- | --- | --- | --- |
| A | ResNet34 ImageNet | None | `--pretrained --no-augment` |
| B | ResNet34 random | None | `--no-pretrained --no-augment` |
| C | Same as A | Lee probability 0.5; shared gain ±2 dB | `--pretrained --no-augment --speckle-filter-prob 0.5 --db-gain-jitter 2` |

Local chip units/processing tags are unset. Unfiltered sigma0 dB is a working
assumption, **not verified provenance**. Our RTC inference stack uses gamma0
power, Lee 5×5, then dB. The optional train-only filter converts dB to power,
reuses the exact Step-6 Lee function and returns to dB. Gain jitter applies one
shared offset to all SAR bands before clipping/normalization, not terrain units;
derived ratios follow the usual clipped VV minus VH rule. It does not perform
terrain/radiometric correction or make sigma0 equivalent to gamma0. No network
lookup was used to establish a processing history. Whether these options improve
transfer must be measured. Prediction receipts flag this unvalidated domain gap.
`--add-ratio` appends the supported VV-minus-VH channel(s) to the configured SAR
inputs and stores their own training constants and actual channel order.

## Kaggle GPU notebook

Enable a GPU accelerator. Internet is needed for your clone/install and the
optional ImageNet download. Upload a Kaggle dataset containing `S1Hand`,
`LabelHand`, `splits`, and optionally `stats.json`; no automatic dataset download
is implemented. Replace the placeholders below with your repository and dataset
mount. Kaggle mounts are read-only: copy the files to `/kaggle/working` so missing
or stale TRAIN statistics can be written. The metadata GeoJSON is optional for
training.

Cell 1 — clone and install:

```python
import subprocess

REPO_URL = "https://github.com/<owner>/floodintel.git"  # Replace this.
subprocess.run(["git", "clone", REPO_URL, "/kaggle/working/floodintel"], check=True)
subprocess.run(["python", "-m", "pip", "install", "-r",
                "/kaggle/working/floodintel/requirements.txt"], check=True)
```

Cell 2 — locate the uploaded data and make a writable copy:

```python
from pathlib import Path
import shutil

UPLOADED = Path("/kaggle/input/<dataset-slug>/sen1floods11")  # Replace this.
DATA_ROOT = Path("/kaggle/working/sen1floods11")
for folder in ("S1Hand", "LabelHand", "splits"):
    shutil.copytree(UPLOADED / folder, DATA_ROOT / folder, dirs_exist_ok=True)
if (UPLOADED / "stats.json").exists():
    shutil.copy2(UPLOADED / "stats.json", DATA_ROOT / "stats.json")
```

Cell 3 — full experiment A (change the flags for B/C):

```python
OUT_DIR = "/kaggle/working/runs/a_resnet34"
subprocess.run([
    "python", "-m", "src.model.train",
    "--data-root", str(DATA_ROOT), "--out-dir", OUT_DIR,
    "--epochs", "30", "--batch-size", "8", "--lr", "0.001",
    "--crop-size", "256", "--encoder", "resnet34", "--backend", "smp",
    "--pretrained", "--no-augment", "--channels", "post_vv", "post_vh",
    "--seed", "0", "--num-workers", "0",
], cwd="/kaggle/working/floodintel", check=True)
```

Reduce batch size if GPU memory is insufficient. Stats signatures include file
locations, so copied stale stats are regenerated from TRAIN at the new location.
Never select the experiment or its probability threshold using Bolivia/test.

Cell 4 — download the best checkpoint and retain the evidence:

```python
from IPython.display import FileLink, display
display(FileLink(f"{OUT_DIR}/best.pt"))
display(FileLink(f"{OUT_DIR}/metrics.json"))
display(FileLink(f"{OUT_DIR}/history.jsonl"))
```

Alternatively download them from Kaggle's Output panel. `best.pt` contains the
actual ModelConfig/backend, channel order, normalization constants, train-derived
positive weight, selected threshold, args, seed, split sizes/regions, library
versions and best-effort Git commit. `last.pt` also supports continuation through
optimizer, scheduler, scaler and RNG states. Neither embeds an arbitrary Python
model object; loading uses tensor/primitive-only checkpoint decoding.

## Selection, resume and standalone scoring

AdamW uses stepwise linear warmup followed by cosine decay and gradient clipping.
`best.pt` is selected by VAL water IoU at threshold 0.5. For that checkpoint the
VAL best-F1 threshold is stored, then fixed for reporting val, test and Bolivia.
Val is used for both selection and threshold tuning; it is not an independent
generalization estimate. Empty metric denominators score zero; a split with no
valid pixels reports undefined scores rather than invented performance. Early
stopping uses `--patience` epochs without improved selection IoU.

Resume with the same data, channels, backend dimensions, seed and augmentation
settings. `--epochs` is the **total target**, not additional epochs. Extending it
extends the cosine schedule; it is not the same trajectory as a run originally
planned with that longer budget. Epoch-boundary RNG continuation is exact with
`num_workers=0`; worker-local states are not recoverable for multiworker loading.
The original `best.pt` must remain next to a `last.pt` being resumed.

```sh
.venv/bin/python -m src.model.train --data-root data/external/sen1floods11 --out-dir runs/local_a --epochs 60 --batch-size 2 --crop-size 256 --encoder resnet34 --pretrained --no-augment --seed 0 --resume runs/local_a/last.pt
.venv/bin/python -m src.model.evaluate --checkpoint runs/local_a/best.pt --data-root data/external/sen1floods11 --split test
.venv/bin/python -m src.model.evaluate --checkpoint runs/local_a/best.pt --data-root data/external/sen1floods11 --split bolivia
```

Scoring prints the metrics, per-region table and ten-bin reliability data and
writes `metrics_test.json` or `metrics_bolivia.json`. It uses the exact same
validation function as training, the checkpoint constants and stored threshold.
The runner's `metrics.json` retains separate val/test/Bolivia entries and caveats.
No threshold is chosen on test or Bolivia.

## App inference

```python
from src.model.predict import predict_incident

probability, valid_mask, metadata = predict_incident(
    incident_or_stack, "runs/local_a/best.pt", quicklook="water_preview.png"
)
```

The two-band model selects post-VV/post-VH by name from the four-band stack.
Four-band models require matching pre-event training data from a later adapter;
Sen1Floods11 cannot supply those bands. Missing terrain bands also raise, rather
than being synthesized. Checkpoint constants own clipping/normalization; the
stack's common validity mask is respected, invalid pixels are zero in the input
and NaN in the probability map. Receipts include channel differences, source
warnings and low-valid-fraction warnings. The stored threshold is returned,
without binarizing or choosing a new one.

Default tiles have 512-pixel output cores, 128-pixel overlap, 64-pixel context
halos and bounded batches of two 640×640 patches. Reflect-padding handles global
edges; singleton dimensions use edge padding. Cores use smooth Hann blending.
These defaults align the sampling grid to the encoder stride. Very large
receptive fields and spatial normalization can still be tile-dependent; testing
a local translation-equivariant convolution does not prove every U-Net equals
full-image inference. `tta=True` enables deterministic horizontal/vertical flip
averaging. The overlay is a visual water-probability check, not damage evidence.
