# tusi_cnn

PyTorch research project evaluating a **Tusi-Couple geometric pre-filtering pipeline**
for mammography classification (malignant vs. benign). The core question: does
transforming 2D spatial image patches into multi-angle, phase-delayed 1D harmonic
diameter sweeps improve feature extraction for radial/spicular structures compared to
raw 2D CNN input?

## Task Goal

Compare a baseline 2D CNN against the same architecture fed Tusi-filtered input, on
identical data splits and hyperparameters, and report whether the geometric
pre-filter measurably helps detect spiculated (malignant-pattern) masses vs. smooth
(benign-pattern) ones.

## Architecture / Module Layout

Keep the code modular — one concern per file, all under a flat top-level package
(no src/ nesting needed for a project this size):

- `dataset.py` — Dataset loading and preprocessing.
  - Prefer a synthetic dataset generator first (dense circular "lumps" with and
    without radiating spicules) for fast iteration, since it gives ground-truth
    control over the radial structure the Tusi filter is meant to exploit.
  - Leave a clean seam to swap in CBIS-DDSM (or another public mammography dataset)
    later without touching model/training code.
  - Standard preprocessing: resize, intensity normalization, ROI cropping centered
    on the lesion/lump.
  - Return `(image_tensor, label)` pairs; label is binary (malignant=1, benign=0).

- `tusi_filter.py` — `TusiRadialFilter` module/transform (the experimental core).
  - Input: a 2D image crop centered on an ROI.
  - For `N` evenly spaced angles `theta_i = i * pi / N`, extract a diameter intensity
    profile through the ROI center at that angle.
  - Apply phase-delayed 1D harmonic sampling along each diameter (i.e., sample the
    diameter profile with a per-angle phase offset derived from the Tusi-couple
    construction) to build a structured 2D feature map of shape
    `(N angles, D radial samples)`.
  - Implement as a `torch.nn.Module` (or a callable transform) so it composes with
    `torchvision.transforms` and can run on batches / GPU.
  - Must be differentiable or at least side-effect-free w.r.t. autograd if it ever
    sits inside the model rather than purely in the data pipeline — default to
    treating it as a preprocessing transform (no grad needed) unless there's a
    reason to learn its parameters.
  - Document the geometric derivation (Tusi couple: a small circle rolling inside a
    larger circle traces a diameter — that's the basis for the phase-delayed radial
    sampling) in the module docstring so the math is traceable.

- `models.py` — Model definitions.
  - Baseline: standard 2D CNN (ResNet18 via `torchvision.models`, or a lightweight
    custom CNN if training from scratch is preferred for a small synthetic dataset).
  - Tusi branch: same backbone family, adapted for `(N, D)` structured input (treat
    angle axis and radial axis analogously to spatial axes, or reshape as needed —
    keep the backbone otherwise identical to the baseline so the comparison isolates
    the input representation, not the architecture).

- `train.py` — Training loop.
  - Train baseline and Tusi-filtered models on identical train/test splits with
    identical hyperparameters (optimizer, learning rate, epochs, loss function —
    BCE/BCEWithLogits for binary classification).
  - Log per-epoch loss/accuracy for both runs so training curves are comparable.

- `evaluate.py` — Metrics and plots.
  - Compute Accuracy, ROC-AUC, Sensitivity/Recall, F1-Score for both models on the
    held-out test set.
  - Plot side-by-side training curves and ROC curves (baseline vs. Tusi) to compare
    convergence speed and performance.

## Conventions

- Every class and function gets a docstring — this is an experimental/research repo
  where the *why* (esp. the geometric reasoning in `tusi_filter.py`) matters as much
  as the *what*.
- Use Python's `logging` module (not bare `print`) for training/eval progress.
- Keep baseline and Tusi runs sharing one training/eval codepath (parametrize by
  model + input transform) rather than duplicating the loop — avoids drift between
  the two conditions that would invalidate the comparison.
- No CLAUDE.md-documented dependency on CBIS-DDSM download/licensing steps until the
  synthetic pipeline is working end-to-end; treat real-data integration as a later,
  separate step.

## Status

Full pipeline built, tested, and run end-to-end on real data. Currently running a
**second** experiment (polyp detection) to test the core Tusi hypothesis on a
cleaner, less ambiguous task. Read this whole section before doing anything else.

### Files that exist and what they do

Core 5 (per architecture above), all written and working:
- `dataset.py` — `SyntheticLumpDataset` (synthetic, still works), `CBISDDSMDataset`
  (thin `ImageFolder` wrapper over `organized/`), `default_transform()` (baseline:
  resize+grayscale→3ch+normalize), `tusi_transform()` (baseline transform +
  `TusiRadialFilter` + 3ch replication).
- `tusi_filter.py` — `TusiRadialFilter(n_angles=48, n_radial=112)`, vectorized via
  `F.grid_sample`, batched/GPU-capable. Verified against a manual reference
  implementation.
- `models.py` — `TusiCompatibleCNN` (the shared backbone used for the fair
  baseline-vs-Tusi comparison; ~390K params, works on both input shapes via
  `AdaptiveAvgPool2d`). Also `TusiAwareCNN` (2D, circular-padded angle axis) and
  `TusiSignalCNN` (1D-signal, per-angle `Conv1d` + circular cross-angle
  aggregation) — both **exploratory only**, not part of the fair comparison.
- `train.py` — `run_training(branch, ...)` shared by both branches;
  `BRANCHES = ("baseline", "tusi")`; saves to `runs/`.
- `evaluate.py` — accuracy/ROC-AUC/sensitivity/F1 + training-curve and ROC plots;
  reads `runs/`.

Extra (built during exploration, not in the original 5-file spec):
- `organize_dataset.py` — **one-time prep script**, run before `dataset.py`'s
  `CBISDDSMDataset` works. Reads raw CBIS-DDSM CSVs, resolves the Kaggle mirror's
  broken `.dcm`→`.jpg` paths (see its docstring — join on `PatientID` + role from
  `dicom_info.csv`, NOT the series-UID folder name, which doesn't reliably match),
  crops each MLO-view mass lesion around its mask centroid, writes
  `organized/{train,test}/{benign,malignant}/*.jpg`. Mask is used only to find the
  crop center — never saved, never seen again after that.
- `explore_tusi_architecture.py` — trains `TusiAwareCNN` or `TusiSignalCNN`
  (`--architecture` flag) with circular angle-roll augmentation, more epochs, weight
  decay. Tusi-only, not a fair comparison, just testing the representation's ceiling.
- `organize_kvasir_dataset.py`, `train_kvasir.py`, `evaluate_kvasir.py` — the new
  polyp experiment, see below.

### Data on disk

- `data/` — extracted CBIS-DDSM (from `archive (4).zip`, 5.3GB, kept in project root).
- `organized/{train,test}/{benign,malignant}/` — 912 MLO-mass crops (711 train / 201
  test), built by `organize_dataset.py`. **This is what `train.py` actually reads.**
- `kvasir_data/Kvasir-SEG/{images,masks}/` — extracted from `kvasir-seg.zip` (46MB,
  in project root). 1000 polyp images + matching masks.
- `kvasir_data/hyperkvasir_normal/` — 1400 normal-tissue images (`cecum` +
  `retroflex-rectum` only — deliberately lower-GI to match where the polyps are;
  do NOT add `pylorus`, that's upper-GI/stomach and would let a model shortcut on
  organ type instead of polyp-vs-not). Selectively extracted from
  `C:\Users\HP\Downloads\hyper-kvasir-labeled-images.zip` (3.9GB, NOT copied into
  the project — too big, re-extract selectively from Downloads if needed again).
- `organized_kvasir/{train,test}/{normal,polyp}/` — 2400 crops (1920 train / 480
  test) built by `organize_kvasir_dataset.py`. **v2 / fixed version**: crop box
  size is now adaptive (mask bounding box x 1.6 margin, then resized down to
  224x224) instead of a fixed 224px window. The first version used a fixed
  224px crop and had a real bug — verified 20.3% of polyp crops were entirely
  filled by the polyp (no boundary visible) and 46.4% were cut off at the edge,
  same class of bug as the mammography pipeline's original fixed-crop issue.
  After the fix: 0% entirely filled, 79.9% show the full polyp with clean
  boundary, only 20.1% still cut off (only when polyp+margin exceeds the source
  image's own resolution — unavoidable). Normal crops draw their box size from
  the *same distribution* as the polyp crops' sizes (not a fixed size), and are
  centered on a random point in visible tissue (not image-center) — both
  deliberate, to prevent the model from learning a positional/scale shortcut
  instead of real tissue content. If `organized_kvasir/` needs rebuilding, just
  rerun `python organize_kvasir_dataset.py` — it's fast (~1min).
- `runs/` — mammography experiment artifacts (models, histories, metrics.json,
  plots) for `baseline`, `tusi`, plus exploratory `tusi_aware_cnn` and
  `tusi_signal_cnn` variants.
- `runs_kvasir/` — polyp experiment artifacts, same shape as `runs/`. **Done,
  concluded** — see result below.
- `sample_images/` and `tusi_demo/` — kept-on-purpose visualizations (mask
  centroid/crop verification, Tusi transform walkthrough images), not scratch.

### Mammography experiment — result (concluded)

Baseline vs. Tusi, both `TusiCompatibleCNN`, 20 epochs, identical hyperparameters:

| Model | Accuracy | ROC-AUC | Sensitivity | F1 |
|---|---|---|---|---|
| Baseline (raw pixels) | 0.632 | 0.626 | 0.237 | 0.339 |
| Tusi, plain shared CNN | 0.617 | 0.627 | 0.375 | 0.438 |
| Tusi, 2D circular-pad CNN (exploratory) | 0.577 | 0.632 | 0.388 | 0.422 |
| Tusi, 1D signal CNN (exploratory) | 0.617 | 0.615 | 0.350 | 0.421 |

**Conclusion so far**: ROC-AUC is a statistical tie across every architecture tried
(~0.62-0.63) — no strong evidence Tusi separates the classes better overall. But
Tusi consistently improves sensitivity (catches more actual malignant cases) at
every architecture variant, which is the clinically important number. Absolute
performance is weak for both branches (dataset is small — 711 train — and CBIS-DDSM
is a biopsy-selected population, meaning it's inherently enriched for cases where
image appearance and biopsy truth diverge, since obvious cases don't get biopsied).
Published literature on this same dataset reaches AUC 0.84-0.89 with pretrained
transfer learning + augmentation, which we have not tried — our from-scratch small
CNN is not using techniques known to help.

**Not yet done, flagged as valuable**: a center-jitter sensitivity test — perturb
the mask-derived crop center by a few pixels and see if Tusi's predictions/accuracy
collapse. This would distinguish "the representation doesn't help" from "the
centroid-based centering is unstable / `grid_sample`'s bilinear interpolation blurs
the fine edge detail Tusi depends on" — a real, cheap, unresolved question.

### Why the second (polyp) experiment exists

User's own diagnosis (validated): CBIS-DDSM's core limitation isn't fixable by
architecture — malignant/benign ground truth comes from biopsy, which can diverge
from visual appearance (we found concrete examples: a benign case with spiculated/
jagged appearance, a malignant case with smooth round appearance). Any purely
visual method, Tusi included, is capped by this. Kvasir-SEG (polyp) sidesteps it:
polyp presence is confirmed by the endoscopist's direct visual annotation — no
separate test that can disagree with appearance. Cleaner test of whether Tusi's
geometric transform helps a CNN extract shape information at all, independent of
the medical-ambiguity confound. Caveat this is *not* malignant/benign classification
— it's "is there a polyp," a detection-shaped task, not a diagnostic one.

### Polyp experiment — result (concluded)

Baseline vs. Tusi, both `TusiCompatibleCNN`, 20 epochs, identical hyperparameters,
run on the crop-size-corrected `organized_kvasir/` (see above — first version had a
crop-too-small bug, fixed before this run):

| Model | Accuracy | ROC-AUC | Sensitivity | F1 |
|---|---|---|---|---|
| Baseline (raw pixels) | 0.848 | **0.970** | 0.670 | 0.786 |
| Tusi | **0.865** | 0.954 | **0.705** | **0.813** |

**Conclusion**: both models are far stronger here than on mammography (AUC
0.95-0.97 vs ~0.62) — confirms the "cleaner task" hypothesis: when appearance
reliably matches ground truth (no biopsy divergence), absolute performance jumps a
lot for both representations. Tusi wins 3 of 4 metrics (accuracy, sensitivity, F1);
baseline has a narrow AUC edge, but both AUCs are excellent. This is a more
decisive, more positive result than mammography's near-total tie. Training curves
(`runs_kvasir/training_curves.png`) show real learning (steadily decreasing loss)
for both, unlike mammography's flat plateau — Tusi's train loss drops even lower
than baseline's.

### Immediate next step for a new session

Both experiments (mammography, polyp) are concluded and their results are recorded
above. Nothing is currently running. Options the user hasn't yet chosen between:
1. Center-jitter sensitivity test on the mammography pipeline (still not done —
   perturb the mask-derived crop center by a few pixels, see if Tusi's predictions
   collapse; distinguishes "representation doesn't help" from "centering/
   interpolation is the real problem").
2. LIDC-IDRI spiculation-rating task (predict the radiologist's own perception
   rating, not diagnosis — cleaner isolation of the core geometric hypothesis).
3. Something else (try pretrained transfer learning on mammography baseline to
   approach published 0.84-0.89 AUC; try the polyp task with the exploratory
   architectures; extend polyp experiment with more data; etc.)

Ask the user before picking a direction — do not assume.
