# fixable_feature.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# --------------------------
# 1) Learnable kernel *per pixel* (space-variant conv)
# --------------------------
class DynamicPerPixelConv2d(nn.Module):
    """
    Space-variant convolution. For every spatial location (h,w), predict its own KxK kernel
    (and optional bias) from the input and apply it to the KxK neighborhood around (h,w).

    Args:
        in_ch:   input channels
        out_ch:  output channels
        kernel_size: int or (kH, kW)
        hidden:  hidden channels for kernel predictor MLP (1x1 conv stack)
        stride, padding, dilation: applied to both the unfold op and the effective conv
        groups:  channel groups for the EFFECTIVE conv (prediction still uses all channels)
        bias:    include a dynamic bias term per-pixel
        norm_pred: if True, L2-normalize the predicted kernels over the K*K*Cin/groups axis
                   (stabilizes training; acts like attention-style normalization)
    Shapes:
        x: (B, Cin, H, W)  -> y: (B, Cout, H_out, W_out) (same as a normal conv with given stride/pad/dilation)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int | Tuple[int, int],
        hidden: int = 0,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm_pred: bool = True,
    ):
        super().__init__()
        assert in_ch % groups == 0, "in_ch must be divisible by groups"
        self.in_ch, self.out_ch, self.groups = in_ch, out_ch, groups
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        self.kH, self.kW = kH, kW

        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.norm_pred = norm_pred
        self.use_bias = bias

        pred_in = in_ch
        pred_out = (out_ch * (in_ch // groups) * kH * kW) + (out_ch if bias else 0)

        if hidden and hidden > 0:
            self.pred = nn.Sequential(
                nn.Conv2d(pred_in, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, pred_out, kernel_size=1, bias=True),
            )
        else:
            self.pred = nn.Conv2d(pred_in, pred_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        # Unfold input into (B, Cin*kH*kW, L) where L = H_out*W_out
        patches = F.unfold(
            x,
            kernel_size=(self.kH, self.kW),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )  # (B, Cin*k*k, L)

        # Predict per-location kernels (and optional bias) of shape (B, PRED, H_out, W_out)
        pred = self.pred(x if (self.stride == 1 and self.padding == 0 and self.dilation == 1) else
                         F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, ceil_mode=False))
        # The predictor runs at output resolution. Compute H_out,W_out from patches:
        L = patches.shape[-1]
        H_out = int((H + 2*self.padding - self.dilation*(self.kH-1) - 1) / self.stride + 1)
        W_out = int((W + 2*self.padding - self.dilation*(self.kW-1) - 1) / self.stride + 1)
        assert H_out*W_out == L, "Output size mismatch"

        if self.use_bias:
            dyn_w, dyn_b = torch.split(pred, [self.out_ch * (self.in_ch // self.groups) * self.kH * self.kW, self.out_ch], dim=1)
        else:
            dyn_w, dyn_b = pred, None

        # Reshape predicted weights to (B, Cout, Cin/groups, kH*kW, L)
        dyn_w = dyn_w.view(B, self.out_ch, (self.in_ch // self.groups), self.kH * self.kW, H_out * W_out)

        if self.norm_pred:
            # Normalize across the (Cin/groups * k*k) axis to stabilize
            dyn_w = F.normalize(dyn_w, dim=2)  # across Cin/groups
            dyn_w = F.normalize(dyn_w, dim=3)  # across k*k

        # Reshape patches to (B, groups, Cin/groups, kH*kW, L)
        patches = patches.view(B, self.groups, (self.in_ch // self.groups), self.kH * self.kW, L)

        # Multiply and sum: for each group, do (Cin/groups * k*k) reduction → (B, groups, Cout/groups?, L)
        # We broadcast dyn_w across groups by splitting Cout into groups evenly if desired.
        if self.groups == 1:
            # Reshape patches to (B, Cin, k*k, L) to match the einsum equation.
            patches = patches.view(B, self.in_ch, self.kH * self.kW, L)
            
            # Correct the einsum to sum over channels 'c' and kernel 'k'.
            out = torch.einsum('bockl,bckl->bol', dyn_w, patches) # (B, Cout, L)
        else:
            # Split Cout into groups equally
            assert self.out_ch % self.groups == 0, "out_ch must be divisible by groups"
            dyn_w = dyn_w.view(B, self.groups, self.out_ch // self.groups, (self.in_ch // self.groups), self.kH * self.kW, L)
            out = torch.einsum('bgoikl,bgikl->bgol', dyn_w, patches)  # (B, groups, Cout/groups, L)
            out = out.reshape(B, self.out_ch, L)

        if self.use_bias:
            # dyn_b: (B, Cout, H_out, W_out) → (B, Cout, L)
            dyn_b = dyn_b.view(B, self.out_ch, H_out * W_out)
            out = out + dyn_b

        # Fold back to images
        out = out.view(B, self.out_ch, H_out, W_out)
        return out


# --------------------------
# 2) Attention "fix-feature" (pixels talk to pixels)
#    Lightweight Non-Local Block with residual
# --------------------------
class FixFeatureAttention(nn.Module):
    """
    Non-local (self-attention) over HW tokens with channel reduction.
    Designed to 'fix/align' features by letting every pixel attend to every other pixel.
    O(HW^2), use on moderate maps or after downsampling.

    Args:
      ch:        input/output channels
      heads:     multi-heads
      reduction: bottleneck for qkv (ch // reduction)
      attn_drop, proj_drop: dropout
      use_pos:   add a learnable 2D relative positional bias (small and stable)
    """
    def __init__(self, ch: int, heads: int = 4, reduction: int = 2,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, use_pos: bool = True):
        super().__init__()
        assert ch % heads == 0, "ch must be divisible by heads"
        inner = max(ch // reduction, heads)
        self.heads = heads
        self.scale = (inner // heads) ** -0.5
        self.qkv = nn.Conv2d(ch, inner * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(inner, ch, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout2d(proj_drop)
        self.use_pos = use_pos
        if use_pos:
            # tiny relative bias over a coarse grid that we interpolate to HxW
            self.pos_bias = nn.Parameter(torch.zeros(1, heads, 16, 16))  # (1,H,Ph,Pw)

        self.norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)  # (B, 3*inner, H, W)
        inner = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, [inner, inner, inner], dim=1)

        # reshape to heads
        def reshape_heads(t):
            t = t.view(B, self.heads, inner // self.heads, H * W)  # (B, h, d, N)
            return t

        q = reshape_heads(q)  # (B,h,d,N)
        k = reshape_heads(k)  # (B,h,d,N)
        v = reshape_heads(v)  # (B,h,d,N)

        attn = torch.einsum('bhdi,bhdj->bhij', q * self.scale, k)  # (B,h,N,N)

        if self.use_pos:
            # interpolate pos bias to HxW grid and add (per head)
            pos = F.interpolate(self.pos_bias, size=(H, W), mode='bilinear', align_corners=False)  # (1,h,H,W)
            pos = pos.view(1, self.heads, H * W)  # (1,h,N)
            attn = attn + pos.unsqueeze(-1) + pos.unsqueeze(-2)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.einsum('bhij,bhdj->bhdi', attn, v)  # (B,h,d,N)
        y = y.reshape(B, inner, H, W)
        y = self.proj_drop(self.proj(y))              # (B,C,H,W)

        # channel-wise LN over residual
        y = self.norm((x + y).permute(0,2,3,1)).permute(0,3,1,2)
        return y


# --------------------------
# 3) Full block: Dynamic kernel → attention-fix → MLP head
# --------------------------
class FixableFeatureBlock(nn.Module):
    """
    A practical module that mirrors the diagram:
      Input → DynamicPerPixelConv2d → BN+GELU → FixFeatureAttention → BN+GELU → 1x1 Conv → Out (+residual)

    Set downsample=True to halve H,W before attention (cheaper) and upsample back.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, hidden_pred: int = 0,
                 heads: int = 4, reduction: int = 2, downsample: bool = False):
        super().__init__()
        self.downsample = downsample

        self.dynamic = DynamicPerPixelConv2d(
            in_ch, out_ch, kernel_size=k, hidden=hidden_pred,
            padding=k//2, stride=1, dilation=1, groups=1, bias=True, norm_pred=True
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.GELU()

        self.pool = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.attn = FixFeatureAttention(out_ch, heads=heads, reduction=reduction, attn_drop=0.0, proj_drop=0.0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if downsample else nn.Identity()

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.GELU()
        self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)

        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = self.shortcut(x)
        y = self.dynamic(x)
        y = self.act1(self.bn1(y))

        y = self.pool(y)
        y = self.attn(y)
        y = self.up(y)

        y = self.act2(self.bn2(y))
        y = self.proj(y)
        return y + resid


# --------------------------
# Tiny sanity check
# --------------------------
if __name__ == "__main__":
    x = torch.randn(2, 32, 32, 32)  # (B,C,H,W)
    block = FixableFeatureBlock(in_ch=32, out_ch=64, k=3, hidden_pred=64, heads=4, reduction=2, downsample=True)
    y = block(x)
    print("out:", y.shape)  # expected: (2, 64, 32, 32)