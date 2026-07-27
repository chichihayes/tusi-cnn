# Tusi-Couple Radial Filter for Medical Image Classification

A PyTorch research project testing whether a geometric pre-filter — inspired by
the **Tusi couple** (a small circle rolling inside a larger circle) — helps a
CNN classify medical images better than raw pixels do.

**Full narrated walkthrough with all results embedded: [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)**
— open it directly on GitHub to see everything without running anything, or run
it yourself (locally, VS Code, or Colab) with zero setup beyond `pip install`.

## The idea

For `N` evenly spaced angles around a lesion's center, sample `D` points along
each angle's diameter at harmonic (cosine), phase-shifted positions — derived
from the classical Tusi-couple construction, where a point on a small circle
rolling inside a larger one traces a diameter via `x(phi) = R*cos(phi)`. Stack
the results into an `(N, D)` grid (angle × radius) and feed that to a CNN
instead of the raw image.

| Original crop | Tusi-filtered |
|---|---|
| ![original](tusi_filtered_examples/mammography_tusi_transform_step1_original_crop.png) | ![filtered](tusi_filtered_examples/mammography_tusi_transform_step3_filtered_output.png) |

## Repo structure

Everything is organized by pipeline — pick the folder for the domain you care
about, everything you need is inside it:

```
core/                        Shared code both pipelines depend on
├── tusi_filter.py           TusiRadialFilter — the core transform
├── models.py                TusiCompatibleCNN + exploratory architectures
├── dataset.py                Synthetic dataset + shared transforms
├── training.py               Shared training-loop primitives
└── evaluation.py             Shared metrics + plotting

mammography/                 Mammography pipeline (CBIS-DDSM, malignant vs. benign)
├── organize_dataset.py      One-time crop preprocessing (already run — see dataset/)
├── train.py                 Train baseline + Tusi branches
├── evaluate.py               Compute metrics, produce plots
├── explore_architectures.py Exploratory Tusi-only architecture variants
├── dataset/{train,test}/{benign,malignant}/   The actual crops used (committed)
└── results/                  Trained models, metrics, plots

polyp/                       Polyp pipeline (Kvasir-SEG + HyperKvasir, polyp vs. normal)
├── organize_dataset.py      One-time crop preprocessing (already run — see dataset/)
├── train.py                 Train baseline + Tusi branches
├── evaluate.py               Compute metrics, produce plots
├── dataset/{train,test}/{normal,polyp}/       The actual crops used (committed)
└── results/                  Trained models, metrics, plots

tusi_filtered_examples/      Every example image used in the walkthrough, one
                              folder, consistently named (mammography_*, polyp_*)
notebooks/walkthrough.ipynb  The full story, pre-executed with embedded results
README.md, requirements.txt
```

No file lives outside a folder that explains its purpose. Nothing here needs
downloading — the committed `dataset/` folders in each pipeline are the actual,
small (a few MB), already-cropped images the results below were produced from.

## Results

Two domains, same controlled comparison both times: baseline (raw pixels) vs.
Tusi (radial transform), identical CNN architecture (`TusiCompatibleCNN`,
~390K params — literally the same model both times), identical
hyperparameters. Only the input representation differs.

| Domain | Model | Accuracy | ROC-AUC | Sensitivity | F1 |
|---|---|---|---|---|---|
| Mammography (CBIS-DDSM) | Baseline | 0.632 | 0.626 | 0.237 | 0.339 |
| Mammography (CBIS-DDSM) | Tusi | 0.617 | 0.627 | 0.375 | 0.438 |
| Polyps (Kvasir-SEG) | Baseline | 0.848 | 0.970 | 0.670 | 0.786 |
| Polyps (Kvasir-SEG) | Tusi | 0.865 | 0.954 | 0.705 | 0.813 |

**Key finding**: Tusi consistently improves sensitivity over baseline in both
domains — but absolute performance is dominated by whether the label
genuinely matches visual appearance in a given domain. Mammography labels come
from biopsy, which can diverge from what a lesion looks like; polyp presence
is confirmed by direct visual annotation, no such divergence possible, and
both models score far higher there. Full explanation, with concrete examples
of the appearance-vs-biopsy divergence and the crop-size bug we caught and
fixed mid-project, is in the notebook.

## Quickstart

```bash
git clone https://github.com/chichihayes/tusi-cnn
cd tusi-cnn
pip install -r requirements.txt
```

**To view everything (no execution needed):** open `notebooks/walkthrough.ipynb`
on GitHub, or in VS Code / Jupyter / Colab — all outputs are pre-rendered.

**To re-run training yourself**, from the repo root:
```bash
python mammography/train.py --epochs 20
python mammography/evaluate.py

python polyp/train.py --epochs 20
python polyp/evaluate.py
```
(Each script also works run from inside its own folder, e.g.
`cd mammography && python train.py`.)

## Reproducing from raw data (optional)

The committed `dataset/` folder in each pipeline already contains everything
needed to reproduce the results above — you do not need the raw source
datasets. If you want to rebuild the crops from scratch instead:

- **CBIS-DDSM**: [Kaggle mirror](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) → extract to `mammography/raw_data/` → `python mammography/organize_dataset.py`
- **Kvasir-SEG**: [official download](https://datasets.simula.no/kvasir-seg/) → extract to `polyp/raw_data/Kvasir-SEG/`
- **HyperKvasir normal images**: [official download](https://datasets.simula.no/hyper-kvasir/), labeled-images subset only (`anatomical-landmarks/cecum` + `anatomical-landmarks/retroflex-rectum`) → `polyp/raw_data/hyperkvasir_normal/` → `python polyp/organize_dataset.py`
