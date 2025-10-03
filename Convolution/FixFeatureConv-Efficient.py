import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# --------------------------
# 1) Efficient DynamicPerPixelConv2d with Grouped Prediction
# --------------------------
class DynamicPerPixelConv2d(nn.Module):
    """
    MODIFIED: Added pred_groups to make the kernel prediction itself a grouped convolution,
    and expansion_ratio to create a low-rank bottleneck for the predicted weights.
    This drastically reduces the parameters in the predictor network.
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
        # New parameters for efficiency
        pred_groups: int = 4, # Use grouped conv in the predictor
        expansion_ratio: float = 0.25, # Bottleneck for predicted weights
    ):
        super().__init__()
        assert in_ch % groups == 0 and out_ch % groups == 0, "in_ch and out_ch must be divisible by groups"
        self.in_ch, self.out_ch, self.groups = in_ch, out_ch, groups
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        self.kH, self.kW = kH, kW

        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.norm_pred = norm_pred
        self.use_bias = bias
        
        # --- Efficiency Modification ---
        # The predictor's output channels are now much smaller
        pred_in = in_ch
        self.bottleneck_dim = int(expansion_ratio * out_ch)
        
        # We predict a low-rank representation of the kernel
        # Total channels needed for the low-rank weights and the projection matrix
        pred_out_weights = self.bottleneck_dim * (in_ch // groups) * kH * kW
        pred_out_proj = self.out_ch * self.bottleneck_dim
        
        pred_out = (pred_out_weights + pred_out_proj) // pred_groups # Divide by pred_groups for grouped conv
        pred_out += (out_ch if bias else 0)


        if hidden and hidden > 0:
            self.pred = nn.Sequential(
                nn.Conv2d(pred_in, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                # The predictor is now a grouped convolution
                nn.Conv2d(hidden, pred_out, kernel_size=1, bias=True, groups=pred_groups),
            )
        else:
            self.pred = nn.Conv2d(pred_in, pred_out, kernel_size=1, bias=True, groups=pred_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        patches = F.unfold(
            x, kernel_size=(self.kH, self.kW), dilation=self.dilation,
            padding=self.padding, stride=self.stride
        )
        L = patches.shape[-1]
        H_out = int((H + 2 * self.padding - self.dilation * (self.kH - 1) - 1) / self.stride + 1)
        W_out = int((W + 2 * self.padding - self.dilation * (self.kW - 1) - 1) / self.stride + 1)
        assert H_out * W_out == L, "Output size mismatch"
        
        pred = self.pred(x if (self.stride == 1 and self.padding == 0 and self.dilation == 1) else
                         F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, ceil_mode=False))

        # --- Efficiency Modification: Reconstruct kernel from low-rank representation ---
        pred_weights_size = (self.bottleneck_dim * (self.in_ch // self.groups) * self.kH * self.kW)
        pred_proj_size = (self.out_ch * self.bottleneck_dim)
        
        # We need to reshape the grouped output back to a dense tensor
        pred = pred.view(B, -1, H_out, W_out)

        if self.use_bias:
            pred_w, pred_proj, dyn_b = torch.split(pred, [pred_weights_size, pred_proj_size, self.out_ch], dim=1)
        else:
            pred_w, pred_proj = torch.split(pred, [pred_weights_size, pred_proj_size], dim=1)
            dyn_b = None

        # [B, bottleneck * (Cin/g) * k*k, L]
        pred_w = pred_w.view(B, self.bottleneck_dim, -1, L)
        # [B, Cout * bottleneck, L]
        pred_proj = pred_proj.view(B, self.out_ch, self.bottleneck_dim, L)
        
        # Create the full kernel via einsum: [B, Cout, (Cin/g)*k*k, L]
        dyn_w = torch.einsum('bobl,bikl->bockl', pred_proj, pred_w)

        if self.norm_pred:
            dyn_w = F.normalize(dyn_w, p=2, dim=2)

        patches = patches.view(B, self.groups, (self.in_ch // self.groups), self.kH * self.kW, L)
        patches = patches.view(B, self.groups, -1, L)
        
        # Reshape dyn_w for grouped convolution logic
        dyn_w = dyn_w.view(B, self.groups, self.out_ch // self.groups, (self.in_ch // self.groups) * self.kH * self.kW, L)
        
        out = torch.einsum('bgoil,bgil->bgol', dyn_w, patches)
        out = out.reshape(B, self.out_ch, L)

        if self.use_bias:
            dyn_b = dyn_b.view(B, self.out_ch, L)
            out = out + dyn_b

        return out.view(B, self.out_ch, H_out, W_out)

# --------------------------
# 2) FixFeatureAttention (Unchanged, but we will use a higher reduction factor)
# --------------------------
class FixFeatureAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4, reduction: int = 2,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, use_pos: bool = True):
        super().__init__()
        assert ch % heads == 0
        inner = max(ch // reduction, heads)
        self.heads, self.scale = heads, (inner // heads) ** -0.5
        self.qkv = nn.Conv2d(ch, inner * 3, 1, bias=False)
        self.proj = nn.Conv2d(inner, ch, 1, bias=True)
        self.attn_drop, self.proj_drop, self.use_pos = nn.Dropout(attn_drop), nn.Dropout2d(proj_drop), use_pos
        if use_pos: self.pos_bias = nn.Parameter(torch.zeros(1, heads, 16, 16))
        self.norm = nn.LayerNorm(ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        inner = qkv.shape[1] // 3
        q, k, v = map(lambda t: t.view(B, self.heads, inner // self.heads, H * W), qkv.chunk(3, dim=1))
        attn = torch.einsum('bhdi,bhdj->bhij', q * self.scale, k)
        if self.use_pos:
            pos = F.interpolate(self.pos_bias, size=(H, W), mode='bilinear', align_corners=False)
            attn = attn + pos.view(1, self.heads, -1).unsqueeze(-1) + pos.view(1, self.heads, -1).unsqueeze(-2)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        y = torch.einsum('bhij,bhdj->bhdi', attn, v).reshape(B, inner, H, W)
        y = self.proj_drop(self.proj(y))
        return self.norm((x + y).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# --------------------------
# 3) The EfficientFixableFeatureBlock
# --------------------------
class EfficientFixableFeatureBlock(nn.Module):
    """
    MODIFIED: This block now uses the more efficient DynamicPerPixelConv2d
    and is configured with a higher attention reduction factor.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, hidden_pred: int = 0,
                 heads: int = 4, reduction: int = 4, # Increased reduction from 2 to 4
                 pred_groups: int = 4, expansion_ratio: float = 0.25):
        super().__init__()
        self.dynamic = DynamicPerPixelConv2d(
            in_ch, out_ch, kernel_size=k, hidden=hidden_pred, padding=k // 2,
            norm_pred=True, pred_groups=pred_groups, expansion_ratio=expansion_ratio
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.GELU()
        self.attn = FixFeatureAttention(out_ch, heads=heads, reduction=reduction)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.GELU()
        self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = self.shortcut(x)
        y = self.dynamic(x)
        y = self.act1(self.bn1(y))
        y = self.attn(y)
        y = self.act2(self.bn2(y))
        y = self.proj(y)
        return y + resid

# --------------------------
# Comparison
# --------------------------
if __name__ == "__main__":
    try:
        from thop import profile
    except ImportError:
        print("Please install thop for MACs/parameter counting: pip install thop")
        profile = None

    if profile:
        # Original block from previous prompt
        class FixableFeatureBlock(nn.Module):
            def __init__(self, in_ch, out_ch, k=3, hidden_pred=0, heads=4, reduction=2, downsample=False):
                super().__init__()
                # Using the original DynamicPerPixelConv2d definition for a fair comparison
                class OriginalDynamicConv(nn.Module):
                    def __init__(self, in_ch, out_ch, kernel_size, hidden, padding, stride, dilation, groups, bias, norm_pred):
                        super().__init__()
                        self.in_ch, self.out_ch, self.groups = in_ch, out_ch, groups
                        if isinstance(kernel_size, int): kH = kW = kernel_size
                        else: kH, kW = kernel_size
                        self.kH, self.kW = kH, kW
                        self.stride, self.padding, self.dilation, self.norm_pred, self.use_bias = stride, padding, dilation, norm_pred, bias
                        pred_in = in_ch
                        pred_out = (out_ch * (in_ch // groups) * kH * kW) + (out_ch if bias else 0)
                        if hidden > 0: self.pred = nn.Sequential(nn.Conv2d(pred_in, hidden, 1), nn.GELU(), nn.Conv2d(hidden, pred_out, 1))
                        else: self.pred = nn.Conv2d(pred_in, pred_out, 1)
                    def forward(self, x):
                        B, Cin, H, W = x.shape
                        patches = F.unfold(x, (self.kH, self.kW), self.dilation, self.padding, self.stride)
                        L = patches.shape[-1]
                        H_out = int((H + 2*self.padding - self.dilation*(self.kH-1) - 1)/self.stride + 1)
                        W_out = int((W + 2*self.padding - self.dilation*(self.kW-1) - 1)/self.stride + 1)
                        pred = self.pred(x if (self.stride==1 and self.padding==0) else F.avg_pool2d(x, self.stride))
                        if self.use_bias: dyn_w, dyn_b = torch.split(pred, [self.out_ch*(self.in_ch//self.groups)*self.kH*self.kW, self.out_ch], dim=1)
                        else: dyn_w, dyn_b = pred, None
                        dyn_w = dyn_w.view(B, self.out_ch, (self.in_ch // self.groups), self.kH * self.kW, L)
                        if self.norm_pred: dyn_w = F.normalize(dyn_w, p=2, dim=2)
                        patches = patches.view(B, self.groups, (self.in_ch//self.groups), self.kH*self.kW, L)
                        if self.groups == 1:
                            out = torch.einsum('bockl,bckl->bol', dyn_w.squeeze(1), patches.squeeze(1))
                        else:
                             dyn_w = dyn_w.view(B, self.groups, self.out_ch // self.groups, (self.in_ch // self.groups), self.kH * self.kW, L)
                             out = torch.einsum('bgoikl,bgikl->bgol', dyn_w, patches).reshape(B, self.out_ch, L)
                        if self.use_bias: out = out + dyn_b.view(B, self.out_ch, L)
                        return out.view(B, self.out_ch, H_out, W_out)
                self.dynamic = OriginalDynamicConv(in_ch, out_ch, k, hidden_pred, k//2, 1, 1, 1, True, True)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.act1 = nn.GELU()
                self.attn = FixFeatureAttention(out_ch, heads, reduction)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.act2 = nn.GELU()
                self.proj = nn.Conv2d(out_ch, out_ch, 1)
                self.shortcut = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
            def forward(self, x):
                resid = self.shortcut(x)
                y = self.act1(self.bn1(self.dynamic(x)))
                y = self.act2(self.bn2(self.proj(self.attn(y))))
                return y + resid

        dummy_input = torch.randn(1, 64, 32, 32)
        
        # Original Block (High Parameters)
        original_block = FixableFeatureBlock(in_ch=64, out_ch=64, k=3, hidden_pred=64, reduction=2)
        original_macs, original_params = profile(original_block, inputs=(dummy_input,), verbose=False)

        # Efficient Block (Low Parameters)
        efficient_block = EfficientFixableFeatureBlock(in_ch=64, out_ch=64, k=3, hidden_pred=64, reduction=4, pred_groups=4, expansion_ratio=0.25)
        efficient_macs, efficient_params = profile(efficient_block, inputs=(dummy_input,), verbose=False)
        
        reduction_percentage = ((original_params - efficient_params) / original_params) * 100

        print("--- Parameter Comparison ---")
        print(f"Original Block:  {original_params/1e6:.2f}M parameters, {original_macs/1e9:.2f}G MACs")
        print(f"Efficient Block: {efficient_params/1e6:.2f}M parameters, {efficient_macs/1e9:.2f}G MACs")
        print(f"\nParameter Reduction: {reduction_percentage:.2f}%")