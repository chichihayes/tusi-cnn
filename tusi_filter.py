"""TusiRadialFilter — the experimental geometric pre-filter.

Geometric derivation (Tusi couple)
-----------------------------------
The Tusi couple is a classical construction: a small circle of radius ``r``
rolls, without slipping, inside a larger fixed circle of radius ``R = 2r``.
A point fixed on the rim of the small circle does not trace a circular arc
as it rolls — it traces a straight line segment, exactly the diameter of
the large circle. Parametrizing the roll by the angle ``phi`` swept by the
rolling circle's center, the traced point's position along that diameter is

    x(phi) = R * cos(phi)

i.e. a pure cosine oscillation back and forth along the line, not a
uniform sweep. That non-uniform, harmonic relationship between the roll
angle and the position along the diameter is what this filter reuses as a
sampling scheme: instead of reading pixel intensities at evenly spaced
points along a diameter, it reads them at the positions a Tusi couple
would visit, ``r_j = R * cos(2*pi*j/D + phase)``. That sampling is denser
near the two ends of the diameter (where cosine changes slowly) and
sparser near the center (where it changes quickly).

The ``phase`` offset above is tied to the diameter's own angle ``theta_i``
(each diameter is sampled starting from a different point in its cosine
cycle), coupling angle and radius the way a Tusi couple couples rolling
angle and position — this is the "phase-delayed" part of the sampling.

What the filter does end to end
--------------------------------
Given a square image crop centered on a lesion:

1. Choose ``N`` evenly spaced angles ``theta_i = i * pi / N`` for
   ``i = 0..N-1`` (only 0 to pi is needed: a diameter through the center
   already covers both directions, so pi to 2*pi would just repeat them
   reversed).
2. For each angle, sample ``D`` points along the diameter through the
   image center at that angle, using the phase-delayed harmonic spacing
   described above.
3. Stack the results into an ``(N, D)`` grid: row ``i`` is angle
   ``theta_i``'s intensity profile, column ``j`` is the ``j``-th harmonic
   sample along that diameter.

The output no longer resembles the input image spatially — a spicule
radiating outward from the lesion center, at any orientation, becomes a
similarly-shaped streak along the radius axis of *some* row of this grid,
rather than a rotation-dependent shape in raw x/y pixels.

This is a fixed geometric resampling with no learnable parameters — it is
implemented with ``F.grid_sample`` so it runs batched on GPU and is
technically differentiable, but per the project convention it is used as a
preprocessing transform (no gradient flows through it in practice).
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class TusiRadialFilter(nn.Module):
    """Transform a square image crop into a phase-delayed radial sweep map.

    Callable on a single image or a batch; composes with
    ``torchvision.transforms`` like any other transform.
    """

    def __init__(self, n_angles: int = 48, n_radial: int = 112) -> None:
        """Configure the sweep resolution.

        Args:
            n_angles: number of evenly spaced diameter angles ``N`` sampled
                over ``[0, pi)`` — becomes the output's row count.
            n_radial: number of harmonic samples ``D`` taken along each
                diameter — becomes the output's column count.
        """
        super().__init__()
        self.n_angles = n_angles
        self.n_radial = n_radial
        self.register_buffer("sample_grid", self._build_sample_grid(), persistent=False)

    def _build_sample_grid(self) -> torch.Tensor:
        """Precompute the (angle, radius) -> (x, y) sampling grid.

        Returns:
            A ``(1, n_angles, n_radial, 2)`` tensor of normalized ``[-1, 1]``
            coordinates, ready for ``F.grid_sample``.
        """
        i = torch.arange(self.n_angles, dtype=torch.float32)
        theta = i * torch.pi / self.n_angles  # theta_i = i * pi / N

        j = torch.arange(self.n_radial, dtype=torch.float32)
        phase = theta.unsqueeze(1)  # per-angle phase delay, tied to theta_i
        cycle = 2 * torch.pi * j.unsqueeze(0) / self.n_radial
        r = torch.cos(cycle + phase)  # harmonic (Tusi-couple) sampling, in [-1, 1]

        x = r * torch.cos(theta).unsqueeze(1)
        y = r * torch.sin(theta).unsqueeze(1)
        grid = torch.stack([x, y], dim=-1)  # (n_angles, n_radial, 2)
        return grid.unsqueeze(0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the radial sweep transform.

        Args:
            image: ``(C, H, W)`` or ``(B, C, H, W)`` tensor, square
                (``H == W``), centered on the lesion.

        Returns:
            ``(C, n_angles, n_radial)`` or ``(B, C, n_angles, n_radial)``
            tensor, matching the input's batching.
        """
        single = image.dim() == 3
        x = image.unsqueeze(0) if single else image

        batch_size = x.shape[0]
        grid = self.sample_grid.expand(batch_size, -1, -1, -1)
        out = F.grid_sample(x, grid, align_corners=True, padding_mode="border")

        return out.squeeze(0) if single else out
