import torch.nn as nn
import torch.nn.functional as F

# =========================
class ResidualGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden=3):
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)
        ])
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.gate(x))
        for layer in self.hidden_layers:
            out = F.relu(layer(out))
        out = self.proj(out)
        return out + residual


class ResidualGatedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden=3, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualGate(input_dim, hidden_dim, num_hidden) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)