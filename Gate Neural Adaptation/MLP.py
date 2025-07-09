import torch,time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ----------- Gated MLP Definition ----------- #
class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, gate_penalty=0.1):
        super(GatedMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.gate_penalty = gate_penalty

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.gates.append(nn.Linear(hidden_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        routing_losses = []
        for layer, gate_layer in zip(self.layers, self.gates):
            x = F.relu(layer(x))
            gate = torch.sigmoid(gate_layer(x))  # gate ∈ (0,1)
            routing_losses.append(gate.mean())  # average gate activation
            x = x * gate  # element-wise gating

        out = self.output_layer(x)
        total_routing_loss = torch.stack(routing_losses).mean()
        return out, total_routing_loss
