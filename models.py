"""Model definitions for the baseline vs. Tusi-filtered comparison.

Both branches use the exact same backbone class and hyperparameters — the
only thing that differs between a "baseline" run and a "Tusi" run is which
``dataset.py`` transform produced the input tensor
(:func:`dataset.default_transform` vs. :func:`dataset.tusi_transform`).
Keeping the architecture identical is what makes the comparison isolate the
input representation rather than confounding it with architecture
differences.

A plain ``torchvision.models.resnet18`` is not used as the default here: its
stem plus four stride-2 stages downsample by 32x total, which degenerates a
Tusi map (48 x 112 by default) down to a 1x3-pixel feature map before the
final layer — too little spatial signal left for the angle/radius structure
to matter. :class:`TusiCompatibleCNN` uses fewer downsampling stages and an
adaptive pool, so it handles both the baseline's 224x224 input and the Tusi
branch's smaller, non-square input equally well.
"""

import logging

import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

logger = logging.getLogger(__name__)


class TusiCompatibleCNN(nn.Module):
    """Small CNN backbone shared by the baseline and Tusi branches.

    A stack of conv/batchnorm/relu/maxpool blocks followed by global average
    pooling and a single linear output (one logit, for use with
    ``BCEWithLogitsLoss``). Global average pooling — rather than a flatten
    into a fixed-size linear layer — is what lets the same architecture
    accept both the baseline's square ``224x224`` input and the Tusi
    branch's smaller, non-square ``n_angles x n_radial`` input without any
    shape-specific code.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_blocks: int = 4,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        """Build the backbone.

        Args:
            in_channels: input channel count (3, to match both branches'
                transforms — see ``dataset.py``).
            num_blocks: number of conv/pool blocks; each halves spatial
                resolution, so this must stay small enough that the
                smallest expected input (the Tusi branch's map) doesn't
                downsample to nothing.
            base_channels: output channels of the first block; each
                subsequent block doubles it.
            dropout: dropout probability before the final linear layer.
        """
        super().__init__()
        blocks = []
        channels = in_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2**i)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        """Return one logit per input image, shape ``(batch,)``."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


class TusiAwareCNN(nn.Module):
    """CNN tailored to the Tusi map's periodic angle axis.

    Exploratory architecture — NOT part of the controlled baseline-vs-Tusi
    comparison, since it only makes sense for Tusi input and has no
    equivalent baseline counterpart (see ``explore_tusi_architecture.py``).
    It exists to test the Tusi representation's upside potential, separate
    from the fair, identical-architecture comparison in
    :class:`TusiCompatibleCNN`.

    A Tusi map's rows are angles ``theta_i = i * pi / N`` sweeping from just
    above 0 to just under pi — row 0 and row N-1 are adjacent diameters
    (both near the same orientation), not unrelated edges. A plain
    ``nn.Conv2d`` zero-pads that axis as if it were a hard boundary, which
    throws away that adjacency. This model pads the angle axis circularly
    instead (and the radial axis normally, since 0 and D-1 there really are
    opposite ends of a diameter, not adjacent).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_blocks: int = 4,
        base_channels: int = 32,
        dropout: float = 0.4,
    ) -> None:
        """Build the backbone.

        Args:
            in_channels: input channel count.
            num_blocks: number of conv/pool blocks.
            base_channels: output channels of the first block; each
                subsequent block doubles it.
            dropout: dropout probability before the final linear layer
                (higher than :class:`TusiCompatibleCNN`'s default since this
                model is only ever trained on the smaller Tusi dataset).
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = in_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2**i)
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(channels, out_channels, kernel_size=3, padding=0),
                        "bn": nn.BatchNorm2d(out_channels),
                        "pool": nn.MaxPool2d(2),
                    }
                )
            )
            channels = out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        """Return one logit per input image, shape ``(batch,)``."""
        for block in self.blocks:
            x = F.pad(x, (1, 1, 0, 0), mode="constant", value=0)  # radial axis: zero-pad
            x = F.pad(x, (0, 0, 1, 1), mode="circular")  # angle axis: circular pad
            x = block["conv"](x)
            x = F.relu(block["bn"](x))
            x = block["pool"](x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


class TusiSignalCNN(nn.Module):
    """1D-signal architecture for Tusi maps — no 2D image convolution at all.

    Exploratory architecture — NOT part of the controlled comparison, same
    reasoning as :class:`TusiAwareCNN`.

    A Tusi map's row ``i`` is not a row of pixels; it is the ``i``-th
    diameter's own 1D intensity-vs-radius signal, produced by phase-delayed
    harmonic sampling (see ``tusi_filter.py``). Treating the whole ``(N, D)``
    grid as a 2D image and sliding a 2D kernel over it — as both
    :class:`TusiCompatibleCNN` and :class:`TusiAwareCNN` do — implicitly
    assumes adjacent angle rows are "spatially" related the way pixel rows
    are. This model instead processes each row as its own 1D signal:

    1. A shared 1D CNN (weights tied across all ``N`` angles, like a
       ``Conv1d`` "sliding" the same feature detector over every diameter)
       extracts a feature vector from each angle's radial signal
       independently — the same edge/peak detector should fire the same way
       regardless of which angle it's looking at.
    2. Those ``N`` per-angle feature vectors are then combined across the
       angle axis with a second, circularly-padded 1D convolution (the
       angle axis is genuinely periodic — see :class:`TusiAwareCNN`), so the
       model can pick up on a spicule pattern repeating across several
       adjacent angles, before a final pooling + linear layer.
    """

    def __init__(
        self,
        radial_channels: tuple[int, ...] = (16, 32, 64),
        angle_channels: int = 64,
        dropout: float = 0.4,
    ) -> None:
        """Build the two-stage signal backbone.

        Args:
            radial_channels: output channels of each per-angle 1D conv
                block (stage 1, along the radial axis).
            angle_channels: output channels of the cross-angle 1D conv
                (stage 2, along the angle axis).
            dropout: dropout probability before the final linear layer.
        """
        super().__init__()
        radial_blocks = []
        in_ch = 1  # the 3 replicated channels from dataset.py are identical; use one
        for out_ch in radial_channels:
            radial_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                )
            )
            in_ch = out_ch
        self.radial_net = nn.Sequential(*radial_blocks)
        self.radial_pool = nn.AdaptiveAvgPool1d(1)

        self.angle_conv = nn.Conv1d(in_ch, angle_channels, kernel_size=3, padding=0)
        self.angle_bn = nn.BatchNorm1d(angle_channels)
        self.angle_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(angle_channels, 1),
        )

    def forward(self, x):
        """Return one logit per input image, shape ``(batch,)``.

        Args:
            x: ``(B, C, n_angles, n_radial)`` Tusi map (``C`` is ignored
                beyond its first channel — see class docstring).
        """
        batch_size, _, n_angles, n_radial = x.shape
        x = x[:, :1]  # (B, 1, n_angles, n_radial) — channels are redundant copies

        # Stage 1: each angle's radial signal through a shared 1D conv net.
        signals = x.reshape(batch_size * n_angles, 1, n_radial)
        features = self.radial_net(signals)  # (B*n_angles, C, D')
        features = self.radial_pool(features).squeeze(-1)  # (B*n_angles, C)
        features = features.reshape(batch_size, n_angles, -1).transpose(1, 2)  # (B, C, n_angles)

        # Stage 2: combine across angles, respecting the circular angle axis.
        features = F.pad(features, (1, 1), mode="circular")
        features = self.angle_conv(features)
        features = F.relu(self.angle_bn(features))
        features = self.angle_pool(features)  # (B, angle_channels, 1)

        return self.classifier(features).squeeze(1)


def build_resnet18(pretrained: bool = False) -> nn.Module:
    """Build a ResNet18 adapted for single-logit binary output.

    Provided for the baseline branch only (see module docstring for why it
    is unsuitable for the Tusi branch's small input). Not used by default.

    Args:
        pretrained: whether to load ImageNet-pretrained weights. Only
            meaningful for the baseline branch — Tusi maps are not natural
            images, so pretrained features would not transfer.

    Returns:
        A ``resnet18`` with its final fully-connected layer replaced by a
        single-output linear layer.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def create_model(architecture: str = "cnn") -> nn.Module:
    """Factory for building a model by name.

    Args:
        architecture: ``"cnn"`` for :class:`TusiCompatibleCNN` (default,
            used for both branches), ``"resnet18"`` for
            :func:`build_resnet18` (baseline only), ``"tusi_aware_cnn"`` for
            :class:`TusiAwareCNN` (Tusi-only, exploratory, 2D), or
            ``"tusi_signal_cnn"`` for :class:`TusiSignalCNN` (Tusi-only,
            exploratory, 1D-signal-based).

    Returns:
        The constructed model.
    """
    if architecture == "cnn":
        return TusiCompatibleCNN()
    if architecture == "resnet18":
        return build_resnet18()
    if architecture == "tusi_aware_cnn":
        return TusiAwareCNN()
    if architecture == "tusi_signal_cnn":
        return TusiSignalCNN()
    raise ValueError(f"unknown architecture: {architecture!r}")
