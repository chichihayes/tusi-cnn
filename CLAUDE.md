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

Full pipeline built, tested, both experiments concluded, **repo restructured for
clarity and pushed to GitHub**: https://github.com/chichihayes/tusi-cnn (public).
Read this whole section before doing anything else — the flat single-folder layout
described earlier in this file is **out of date**; the actual structure is below.

### Current repo structure (as pushed to GitHub)

```
core/                   Shared code both pipelines import from
  tusi_filter.py         TusiRadialFilter (n_angles=48, n_radial=112, F.grid_sample-based)
  models.py               TusiCompatibleCNN (shared backbone, ~390K params) +
                           TusiAwareCNN / TusiSignalCNN (exploratory only)
  dataset.py               SyntheticLumpDataset, CBISDDSMDataset, default_transform(),
                           tusi_transform()
  training.py              BRANCHES, get_device(), run_epoch() — shared training loop
  evaluation.py            load_history/model, collect_predictions, compute_metrics,
                           plot_training_curves, plot_roc_curves — shared eval/plots

mammography/            CBIS-DDSM pipeline (malignant vs. benign)
  organize_dataset.py     One-time prep from raw CBIS-DDSM (already run)
  train.py                 build_datasets() + run_training() + main(); imports core.*
  evaluate.py               Thin wrapper around core.evaluation
  explore_architectures.py  Exploratory Tusi-only variants (TusiAwareCNN/TusiSignalCNN)
  dataset/{train,test}/{benign,malignant}/   912 crops, committed (4.5MB)
  results/                  baseline/tusi model+history+metrics+plots;
                           results/exploratory/ has the tusi_aware/tusi_signal results

polyp/                  Kvasir-SEG + HyperKvasir pipeline (polyp vs. normal)
  organize_dataset.py      One-time prep from raw Kvasir-SEG/HyperKvasir (already run)
  train.py, evaluate.py     Same shape as mammography's
  dataset/{train,test}/{normal,polyp}/       2400 crops, committed (23MB)
  results/                  baseline/tusi model+history+metrics+plots

tusi_filtered_examples/  All demo/example images, ONE folder, consistently named
                         (mammography_*, polyp_*) — no scattered sample_images/tusi_demo
notebooks/walkthrough.ipynb   Full narrated walkthrough, PRE-EXECUTED (outputs embedded,
                         renders on GitHub with no setup). Reads from ../mammography/
                         and ../polyp/, not old organized/ or runs/ paths.
```

**Import pattern**: every pipeline script (`mammography/train.py`,
`polyp/evaluate.py`, etc.) does `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
then `from core.dataset import ...` etc. — works whether run as
`python mammography/train.py` from repo root OR `cd mammography && python train.py`.
Verified both ways.

**What got deleted** (superseded, do not look for these — they no longer exist):
the old flat top-level `dataset.py`, `models.py`, `tusi_filter.py`, `train.py`,
`evaluate.py`, `train_kvasir.py`, `evaluate_kvasir.py`, `organize_dataset.py`,
`organize_kvasir_dataset.py`, `explore_tusi_architecture.py`, and the old
`organized/`, `organized_kvasir/`, `runs/`, `runs_kvasir/`, `sample_images/`,
`tusi_demo/` folders. `GEMINI_PROMPT.md` was also removed (was a one-off prompt
for an external AI, not project documentation).

### Raw data on disk (local only, gitignored, NOT in the GitHub repo)

- `data/` — extracted CBIS-DDSM (from `archive (4).zip`, 5.3GB, local project root).
- `kvasir_data/Kvasir-SEG/{images,masks}/` — extracted Kvasir-SEG (from
  `kvasir-seg.zip`, 46MB, local project root).
- `kvasir_data/hyperkvasir_normal/` — 1400 normal images (`cecum` +
  `retroflex-rectum` only — deliberately lower-GI, NOT `pylorus` which is
  upper-GI/stomach and would let a model shortcut on organ type). Selectively
  extracted from `C:\Users\HP\Downloads\hyper-kvasir-labeled-images.zip` (3.9GB,
  never copied into the project).

None of the above is needed to use the repo — `mammography/dataset/` and
`polyp/dataset/` (the actual small processed crops) are committed to GitHub
directly. Raw data is only needed if regenerating crops from scratch via the
`organize_dataset.py` scripts (which now default to `<pipeline>/raw_data/` as
the expected raw-data location — the local `data/`/`kvasir_data/` folders at
repo root are NOT where those scripts look by default anymore; pass `--raw-dir`
explicitly if regenerating from the existing local raw-data folders).

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
(`polyp/results/training_curves.png`) show real learning (steadily decreasing loss)
for both, unlike mammography's flat plateau — Tusi's train loss drops even lower
than baseline's.

### GitHub repo (done)

Public, at https://github.com/chichihayes/tusi-cnn. Contains all code, both
pipelines' committed datasets/results, and the pre-executed
`notebooks/walkthrough.ipynb`. The "Honest limitations" section was deliberately
removed from both the notebook and README per user request (kept in this
CLAUDE.md file only, below, since that's project memory, not public-facing docs).
Repo was restructured (see above) after the initial push specifically so a
professor or new visitor could understand the layout and run any part of it
without confusion — verified end-to-end after the restructure (both
`mammography/evaluate.py` and `polyp/evaluate.py` rerun successfully, identical
metrics to before the restructure).

### Immediate next step for a new session

Both experiments (mammography, polyp) are concluded and their results are recorded
above, and the repo is live on GitHub with a clean structure. Nothing is currently
running. Options the user hasn't yet chosen between:
1. Center-jitter sensitivity test on the mammography pipeline (still not done —
   perturb the mask-derived crop center by a few pixels, see if Tusi's predictions
   collapse; distinguishes "representation doesn't help" from "centering/
   interpolation is the real problem").
2. LIDC-IDRI spiculation-rating task (predict the radiologist's own perception
   rating, not diagnosis — cleaner isolation of the core geometric hypothesis).
3. 5-fold cross-validation on the polyp dataset (for confidence intervals — note:
   needs a proper paired statistical test like DeLong's for AUC comparison, not
   just naive std-dev across folds; also needs real code changes since the
   current pipeline builds one fixed split, not folds).
4. A naive-downsampling baseline (e.g. raw image resized to ~73x73, matching
   Tusi's ~5,376-value pixel budget) to isolate whether Tusi's efficiency comes
   from its specific geometric reorganization or just fewer pixels in general —
   run as a genuine test, not a foregone conclusion (polyps are coarse/large
   features that may survive naive downsampling better than expected).
5. Literature search for prior art ("polar transform CNN medical imaging",
   "log-polar coordinate neural network", "Fourier radial descriptor deep
   learning") to position this work relative to existing polar-transform CNNs.
6. Something else (pretrained transfer learning on mammography baseline to
   approach published 0.84-0.89 AUC; polyp task with exploratory architectures;
   more polyp data; etc.)

Ask the user before picking a direction — do not assume.
