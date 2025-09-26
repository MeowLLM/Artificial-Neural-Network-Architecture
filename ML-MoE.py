import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Safe Division
# =====================================================
def safe_div(numerator, denominator, mode="very_close", eps=1e-6):
    e_const = torch.tensor(torch.e, device=denominator.device, dtype=denominator.dtype)

    if mode == "too_near":
        safe_den = torch.where(torch.abs(denominator) < eps*0.1, e_const, denominator)
    elif mode == "too_close":
        safe_den = torch.where(torch.abs(denominator) < eps, e_const, denominator)
    elif mode in ["close","very_close"]:
        sign = torch.sign(denominator)
        sign = torch.where(sign==0, torch.ones_like(sign), sign)
        factor = 10 if mode=="very_close" else 1
        safe_den = denominator + factor*eps*sign
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return numerator / safe_den


# =====================================================
# Custom Activations
# =====================================================
class X_EReLU(nn.Module):
    def __init__(self, mode="too_close", eps=1e-6):
        super().__init__()
        self.a1 = nn.Parameter(torch.tensor(1.0)); self.a2 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(1.0)); self.b2 = nn.Parameter(torch.tensor(1.0))
        self.c1 = nn.Parameter(torch.tensor(1.0)); self.c2 = nn.Parameter(torch.tensor(1.0))
        self.d1 = nn.Parameter(torch.tensor(1.0)); self.d2 = nn.Parameter(torch.tensor(1.0))
        self.i  = nn.Parameter(torch.tensor(1.0))
        self.mode, self.eps = mode, eps

    def forward(self, x):
        a = safe_div(self.a1,self.a2,self.mode,self.eps)
        b = safe_div(self.b1,self.b2,self.mode,self.eps)
        c = safe_div(self.c1,self.c2,self.mode,self.eps)
        d = safe_div(self.d1,self.d2,self.mode,self.eps)
        numerator = a + b*x - c*torch.abs(x) + d
        return safe_div(numerator, self.i, self.mode, self.eps)

# =====================================================
# Expert Block
# =====================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2):
        super().__init__()
        act = X_EReLU()
        
        layers = []
        for i in range(depth):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < depth - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =====================================================
# Gating Network
# =====================================================
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)  # [batch, num_experts]


# =====================================================
# Mixture of Experts (MoE)
# =====================================================
class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, 
                 expert_depth=2):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, depth=expert_depth)
            for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gate_weights = self.gating(x)                     # [batch, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, output_dim]
        out = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)          # weighted sum
        return out


# =====================================================
# Example Usage
# =====================================================
if __name__ == "__main__":
    batch_size = 8
    input_dim = 16
    hidden_dim = 32
    output_dim = 10
    num_experts = 4

    model = MoE(input_dim, hidden_dim, output_dim, num_experts=num_experts, 
                expert_depth=3, activation="UltraSMHT")
    
    x = torch.randn(batch_size, input_dim)
    y = model(x)

    print("Output shape:", y.shape)  # [8, 10]
