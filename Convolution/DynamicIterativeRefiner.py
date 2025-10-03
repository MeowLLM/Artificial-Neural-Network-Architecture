import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

Pair = Union[int, Tuple[int, int]]

def _to_pair(v: Pair) -> Tuple[int, int]:
    return (v, v) if isinstance(v, int) else v

def _same_padding(kernel_size: Tuple[int,int], dilation: Tuple[int,int]) -> Tuple[int,int]:
    # only correct for odd kernels; we assert that below
    kH, kW = kernel_size
    dH, dW = dilation
    pad_h = ((kH - 1) * dH) // 2
    pad_w = ((kW - 1) * dW) // 2
    return (pad_h, pad_w)

# ---------------------------
# 1) Shared-kernel iterative refiner (fast, simple)
# ---------------------------
class IterativeRefiner(nn.Module):
    """
    Iteratively refines a feature map using a learnable kH×kW convolution
    with residual updates, for T steps.

    - Connections: full kH×kW neighborhood (diagonals included).
    - Weight sharing across steps (same kernel each step).
    - Optional depthwise mode = per-channel kernel (no channel mixing).
    - Keeps spatial size (requires odd kernel sizes).
    """
    def __init__(
        self,
        channels: int,
        steps: int = 5,
        depthwise: bool = True,
        norm: bool = True,
        step_scale: float = 1.0,
        kernel_size: Pair = 3,
        dilation: Pair = 1,
    ):
        super().__init__()
        self.steps = steps
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels) if norm else nn.Identity()

        kH, kW = _to_pair(kernel_size)
        dH, dW = _to_pair(dilation)
        assert kH % 2 == 1 and kW % 2 == 1, "IterativeRefiner: kernel_size must be odd to preserve HxW."
        pad_h, pad_w = _same_padding((kH, kW), (dH, dW))

        if depthwise:
            self.conv = nn.Conv2d(
                channels, channels,
                kernel_size=(kH, kW),
                padding=(pad_h, pad_w),
                dilation=(dH, dW),
                groups=channels,
                bias=True,
            )
        else:
            self.conv = nn.Conv2d(
                channels, channels,
                kernel_size=(kH, kW),
                padding=(pad_h, pad_w),
                dilation=(dH, dW),
                bias=True,
            )

        # small learned step size (stabilizes training)
        self.alpha = nn.Parameter(torch.tensor(step_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.steps):
            y = self.conv(self.norm(h))
            h = h + self.alpha * y
        return h


# ---------------------------
# 2) Dynamic per-pixel kernel refiner (space-variant)
# ---------------------------
class DynamicIterativeRefiner(nn.Module):
    """
    Predicts a kH×kW kernel PER PIXEL (and per channel via depthwise application),
    applies it to the local patch (via unfold), and updates residually.
    Repeats for T steps with shared predictor weights by default.

    Shapes:
      x: (B, C, H, W)
    Predictor outputs: (B, C*(kH*kW)[ + C if bias ], H, W)

    Options:
      - norm_pred: L2-normalize each per-pixel kernel across the kH*kW axis.
      - use_bias: optional per-pixel scalar bias per channel.
      - Keeps spatial size (requires odd kernel sizes).
    """
    def __init__(
        self,
        channels: int,
        hidden: int = None,
        steps: int = 5,
        norm_pred: bool = True,
        use_bias: bool = False,
        step_scale: float = 1.0,
        kernel_size: Pair = 3,
        dilation: Pair = 1,
    ):
        super().__init__()
        self.steps = steps
        self.norm_pred = norm_pred
        self.use_bias = use_bias
        self.alpha = nn.Parameter(torch.tensor(step_scale, dtype=torch.float32))

        kH, kW = _to_pair(kernel_size)
        dH, dW = _to_pair(dilation)
        assert kH % 2 == 1 and kW % 2 == 1, "DynamicIterativeRefiner: kernel_size must be odd to preserve HxW."
        self.kH, self.kW = kH, kW
        self.dH, self.dW = dH, dW
        self.pad_h, self.pad_w = _same_padding((kH, kW), (dH, dW))
        k_elems = kH * kW

        # Kernel predictor: small 1x1 MLP (Conv → GELU → Conv)
        hid = max(16, channels) if hidden is None else hidden
        out_ch = channels * k_elems + (channels if use_bias else 0)
        self.pred = nn.Sequential(
            nn.Conv2d(channels, hid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, out_ch, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = x
        k_elems = self.kH * self.kW

        for _ in range(self.steps):
            k = self.pred(h)  # (B, C*k_elems[+C], H, W)
            if self.use_bias:
                k, b = torch.split(k, [C * k_elems, C], dim=1)  # b: (B, C, H, W)
            else:
                b = None

            # normalize kernels per pixel/channel if requested
            if self.norm_pred:
                k = k.view(B, C, k_elems, H, W)
                k = k / (k.norm(dim=2, keepdim=True) + 1e-6)
                k = k.view(B, C * k_elems, H, W)

            # gather kH×kW patches (depthwise via unfold)
            patches = F.unfold(
                h,
                kernel_size=(self.kH, self.kW),
                padding=(self.pad_h, self.pad_w),
                dilation=(self.dH, self.dW),
            )  # (B, C*k_elems, H*W)
            weights = k.view(B, C * k_elems, H * W)

            # per-pixel weighted sum across the neighborhood, depthwise
            y = (patches * weights).view(B, C, k_elems, H * W).sum(dim=2)  # (B, C, H*W)
            y = y.view(B, C, H, W)

            if b is not None:
                y = y + b

            h = h + self.alpha * y

        return h


# ---------------------------
# Tiny demo (3×3 grid; try non-square kernels)
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    grid = torch.tensor(
        [[[[0.2, 0.3, 0.2],
           [0.8, 0.5, 0.1],
           [0.7, 0.6, 0.3]]]], dtype=torch.float32
    )  # (1,1,3,3)

    # Shared-kernel with 5x3 kernel
    refiner = IterativeRefiner(
        channels=1, steps=4, depthwise=True, norm=True, step_scale=0.5,
        kernel_size=(5, 3), dilation=1
    )
    out_shared = refiner(grid)

    # Dynamic per-pixel with 3x5 kernel
    dyn_refiner = DynamicIterativeRefiner(
        channels=1, steps=4, norm_pred=True, use_bias=False, step_scale=0.5,
        kernel_size=(3, 5), dilation=1
    )
    out_dynamic = dyn_refiner(grid)

    print("Shared-kernel output:\n", out_shared[0, 0])
    print("Dynamic per-pixel output:\n", out_dynamic[0, 0])