# Tusi-Couple Radial Filter for Medical Image Classification

A PyTorch research project testing whether a geometric pre-filter — inspired by
the **Tusi couple** (a small circle rolling inside a larger circle) — helps a
CNN classify medical images better than raw pixels do.

**Full walkthrough with all results and images: [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)**

## The idea

For `N` evenly spaced angles around a lesion's center, sample `D` points along
each angle's diameter at harmonic (cosine), phase-shifted positions — derived
from the classical Tusi-couple construction, where a point on a small circle
rolling inside a larger one traces a diameter via `x(phi) = R*cos(phi)`. Stack
the results into an `(N, D)` grid (angle x radius) and feed that to a CNN
instead of the raw image. The bet: a boundary radiating outward from a
lesion's center — a spiculated tumor margin, a polyp's edge — becomes a
simpler, more consistent pattern in this representation than in raw x/y
pixels, since rotation-dependent structure becomes translation-dependent
structure.

## Results summary

Two domains tested, same controlled comparison both times: baseline (raw
pixels) vs. Tusi (radial transform), identical CNN architecture
(`TusiCompatibleCNN`, ~390K params, literally the same model both times),
identical hyperparameters — only the input representation differs.

| Domain | Model | Accuracy | ROC-AUC | Sensitivity | F1 |
|---|---|---|---|---|---|
| Mammography (CBIS-DDSM) | Baseline | 0.632 | 0.626 | 0.237 | 0.339 |
| Mammography (CBIS-DDSM) | Tusi | 0.617 | 0.627 | 0.375 | 0.438 |
| Polyps (Kvasir-SEG) | Baseline | 0.848 | 0.970 | 0.670 | 0.786 |
| Polyps (Kvasir-SEG) | Tusi | 0.865 | 0.954 | 0.705 | 0.813 |

**Key finding**: Tusi consistently improves sensitivity over baseline in both
domains — but absolute performance is dominated by whether the label
genuinely matches visual appearance in a given domain. Mammography labels come
from biopsy, which can diverge from what a lesion looks like (we found
concrete counterexamples — see the notebook); polyp presence is confirmed by
direct visual annotation, no such divergence possible, and both models score
far higher there (AUC 0.95-0.97 vs ~0.62).

**Efficiency**: both branches use the identical model, so the Tusi branch's
speed advantage (~9x fewer FLOPs, ~5.8x faster on CPU) comes entirely from
using a smaller input resolution — a hyperparameter choice, not an inherent
property of the transform. See the notebook for the honest framing.

## Repo layout

```
dataset.py                    Dataset loading (synthetic + CBIS-DDSM)
tusi_filter.py                TusiRadialFilter — the core transform
models.py                     TusiCompatibleCNN + exploratory architectures
train.py, train_kvasir.py     Training loops (mammography, polyp)
evaluate.py, evaluate_kvasir.py   Metrics + plots
organize_dataset.py           One-time CBIS-DDSM crop preprocessing
organize_kvasir_dataset.py    One-time polyp/normal crop preprocessing
explore_tusi_architecture.py  Exploratory Tusi-only architecture variants
organized/, organized_kvasir/ The actual crops used to produce the results
                               above (small, committed — nothing needs
                               downloading to explore these)
runs/, runs_kvasir/            Trained model weights, metrics, plots
sample_images/, tusi_demo/     Visualizations used during development
notebooks/walkthrough.ipynb    Full narrated walkthrough with embedded results
```

## Quickstart

```bash
pip install -r requirements.txt

# Re-run the existing results (uses the committed organized/ crops):
python train.py --epochs 20
python evaluate.py

python train_kvasir.py --epochs 20 --out-dir runs_kvasir
python evaluate_kvasir.py --root-dir organized_kvasir --run-dir runs_kvasir
```

## Reproducing from raw data (optional)

`organized/` and `organized_kvasir/` (the actual crops used above) are
already committed — you don't need the raw source datasets to explore the
results or re-run training. If you want to rebuild the crops from scratch:

- **CBIS-DDSM**: [Kaggle mirror](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) → extract to `data/` → `python organize_dataset.py`
- **Kvasir-SEG**: [official download](https://datasets.simula.no/kvasir-seg/) → extract to `kvasir_data/Kvasir-SEG/`
- **HyperKvasir normal images**: [official download](https://datasets.simula.no/hyper-kvasir/), labeled-images subset only (`anatomical-landmarks/cecum` + `anatomical-landmarks/retroflex-rectum`) → `kvasir_data/hyperkvasir_normal/` → `python organize_kvasir_dataset.py`

See `notebooks/walkthrough.ipynb` for the full narrative, including the bugs
we caught and fixed along the way.
