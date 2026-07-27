# Project explanation for Gemini — brick by brick

I'm working on a research project testing a geometric image pre-filter for medical
image classification. I want you to understand the full pipeline, what we built,
what we tested, and what we found — in order, so you have the complete picture.

## 1. The core idea

Take a small image crop centered on something of interest (a lesion, a polyp), and
instead of feeding the raw 2D pixel grid into a CNN, transform it first:

1. Pick N evenly spaced angles around the center, `theta_i = i * pi / N` for
   `i = 0..N-1` (covering 0 to pi is enough — a line through the center already
   covers both directions).
2. For each angle, sample D points along the diameter line through the center at
   that angle — not evenly spaced, but at positions given by
   `r = cos(2*pi*j/D + theta_i)` for `j = 0..D-1`. This harmonic (cosine-based),
   phase-shifted sampling scheme comes from the geometry of a "Tusi couple": a
   small circle of radius r rolling without slipping inside a larger circle of
   radius 2r. A point on the rim of the small circle traces a straight line (a
   diameter of the big circle) as it rolls, and its position along that line is
   exactly `x(phi) = R*cos(phi)` — a pure cosine oscillation, not a linear sweep.
   We reuse that as the sampling scheme, with each diameter's phase tied to its
   own angle.
3. Stack the results into an `(N, D)` grid: row = angle, column = harmonic radial
   sample. This grid, not the original image, is what gets fed to the CNN.

The bet: a boundary/edge radiating outward from the center — like a spiculated
(spiky) tumor margin, or the edge of a round polyp — becomes a simpler, more
consistent pattern in this (angle, radius) representation than it is in raw x/y
pixels, because rotation-dependent structure becomes translation-dependent
structure, which convolutional filters are naturally better at exploiting.

## 2. Implementation (`tusi_filter.py`)

`TusiRadialFilter`, a `torch.nn.Module`. Precomputes the (angle, radius) -> (x, y)
sampling grid once, then uses `F.grid_sample` to bilinearly sample the whole grid
at once — vectorized, batched, GPU-capable. Input `(C, H, W)` or `(B, C, H, W)`,
output `(C, N, D)` or `(B, C, N, D)`.

## 3. Experiment design — the controlled comparison

Two branches, same everything except the input transform:
- **Baseline**: crop -> resize -> grayscale replicated to 3 channels -> normalize
  -> CNN.
- **Tusi**: crop -> resize -> grayscale -> `TusiRadialFilter` -> replicate to 3
  channels -> normalize -> CNN.

Both branches use the **exact same CNN architecture** (`TusiCompatibleCNN`: 4
conv/batchnorm/relu/maxpool blocks, global average pooling so it accepts any input
spatial size, single logit output for `BCEWithLogitsLoss`, ~390K params). Same
optimizer (Adam), same learning rate, same epoch count, same seed for model
initialization. The only variable that differs is the input representation — this
isolates whether the geometric transform helps, rather than confounding it with an
architecture difference.

## 4. First test domain: mammography (CBIS-DDSM)

Public dataset, 1,696 mass lesions, `pathology` label (malignant/benign)
confirmed by biopsy — the clinical gold standard, not a visual read. Built a
one-time preprocessing script that:
- Resolves the dataset's broken file paths (case CSVs point at `.dcm` files that
  don't exist in this Kaggle mirror; had to join on `PatientID` + role via a
  separate `dicom_info.csv` to find the real `.jpg` files).
- Uses each lesion's ROI mask only to find its centroid (never touches the mask
  again after that — the model only ever sees raw cropped tissue).
- Crops a fixed region around that centroid, sorts into
  `{train,test}/{benign,malignant}/` folders.

**Result** (20 epochs, MLO view only, 912 total lesions):

| Model | Accuracy | ROC-AUC | Sensitivity | F1 |
|---|---|---|---|---|
| Baseline | 0.632 | 0.626 | 0.237 | 0.339 |
| Tusi | 0.617 | 0.627 | 0.375 | 0.438 |

ROC-AUC was a statistical tie — no strong evidence Tusi separates the classes
better overall on this dataset. But Tusi consistently improved sensitivity
(catches more actual malignant cases) across every architecture variant we tried
(also tested a circular-padded 2D CNN and a 1D-signal CNN treating each angle as
its own waveform — neither beat the plain shared CNN either).

## 5. Why performance was capped, and the key insight

We diagnosed why: CBIS-DDSM's ground truth comes from biopsy, which can diverge
from what the lesion visually looks like. We found concrete counterexamples in the
data itself: a biopsy-confirmed **benign** lesion with a jagged, spiculated-looking
mask (architectural distortion mimicking cancer), and a biopsy-confirmed
**malignant** lesion with a smooth, round mask (no spikes at all). We also ran a
blind test — looked at 5 raw mammogram crops ourselves, guessed malignant/benign by
eye, checked against the real biopsy labels: 1 out of 5 correct, and the two we
were *most* confident about were both wrong.

This means: no purely visual method — Tusi included — can be expected to
perfectly classify this data, because the label sometimes disagrees with the
image. CBIS-DDSM is also a biopsy-selected population (only ambiguous-enough cases
get biopsied), so it's inherently enriched for exactly the cases where appearance
and truth diverge.

## 6. Second test domain: colon polyps (Kvasir-SEG + HyperKvasir)

To test the core hypothesis on a cleaner task — where visual appearance directly
matches ground truth, no biopsy divergence possible — we switched to polyp
detection. A polyp is a physical bump; its presence is confirmed by direct
endoscopic visual annotation, not a separate test that can disagree with what's
in the image.

- **Positive class**: Kvasir-SEG, 1,000 polyp images with pixel-level masks.
- **Negative class**: HyperKvasir's `normal-cecum` + `normal-retroflex-rectum`
  images (deliberately lower-GI/colon landmarks, matching where the polyps
  actually are — NOT upper-GI landmarks like pylorus, which would let the model
  shortcut on organ type instead of polyp-vs-not).

Cropping approach (after fixing a real bug — see below): crop box size adaptive
to each polyp's own mask bounding box (x1.6 margin) so the full boundary is always
visible, then resized to 224x224. Normal-image crops draw their box size from the
*same distribution* as the polyp crops (not a fixed size) and center on a random
point in visible tissue (not the image center) — both deliberate, to prevent the
model from learning a positional or scale shortcut instead of real tissue content.

**Bug we caught and fixed**: first version used a fixed 224px crop window in
original-image pixels. Measured result: 20.3% of polyp crops were entirely inside
the polyp (no boundary visible at all), 46.4% had the polyp cut off at the crop
edge. Only 33.2% showed a clean full boundary. Fixed by making the crop size
adaptive to each lesion's actual size before resizing down to the model's fixed
input size. After the fix: 0% entirely filled, 79.9% clean full boundary.

**Result** (20 epochs, 2,400 total crops):

| Model | Accuracy | ROC-AUC | Sensitivity | F1 |
|---|---|---|---|---|
| Baseline | 0.848 | 0.970 | 0.670 | 0.786 |
| Tusi | 0.865 | 0.954 | 0.705 | 0.813 |

Both models far stronger here than on mammography (confirms the appearance-vs-truth
divergence was a real factor). Tusi wins 3 of 4 metrics; baseline has a narrow AUC
edge but both AUCs are excellent (>0.95).

## 7. Efficiency finding

Both branches use the identical model (389,633 parameters — literally the same,
by design). But because the Tusi map (48x112 = 5,376 values) is much smaller than
the raw crop (224x224 = 50,176 values), the Tusi branch's forward pass uses ~9.3x
fewer FLOPs (measured: 79.3M vs 740.0M mult-adds) and runs ~5.8x faster on CPU
(4.90ms vs 28.40ms per image, including the transform itself). Important honest
caveat: this isn't because the Tusi *architecture* is inherently more efficient —
it's because we chose a smaller output resolution as a hyperparameter. The
legitimate claim is: at ~9x fewer input values, Tusi still matched or beat the
full-resolution baseline on accuracy/sensitivity/F1 — an information-density
argument, not an architectural-efficiency one.

## 8. Honest limitations, not yet resolved

- Single run per condition, no repeated seeds or cross-validation — results are
  promising, not statistically bulletproof.
- No proper train/validation/test split — model selection uses the last epoch,
  not a validation-selected best epoch, specifically to avoid test-set leakage,
  but this means results are somewhat noisy/epoch-dependent.
- Have not tested whether the mammography pipeline's centroid-based cropping is
  itself stable (center-jitter sensitivity test) — a real, unresolved question
  about whether the low mammography performance is about the representation or
  about implementation fragility (e.g. `grid_sample`'s bilinear interpolation
  blurring the fine edges Tusi depends on).
- Haven't done a literature search to confirm whether this specific Tusi-couple-
  derived harmonic sampling scheme has prior art — general polar-transform CNNs
  exist in the literature (retinal imaging, nodule detection), but this exact
  construction's novelty is unconfirmed.

## What I want from you

[Fill in specifically what you want Gemini to help with — e.g., critique the
experimental design, suggest what to test next, sanity-check the efficiency
framing, help find related work, etc.]
