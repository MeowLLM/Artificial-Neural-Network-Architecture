class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)  # Per-node gating
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        gate_val = torch.sigmoid(self.gate_layer(h1))  # Gate per node
        h1_gated = h1 * gate_val
        out = self.fc2(h1_gated)
        return out, gate_val
