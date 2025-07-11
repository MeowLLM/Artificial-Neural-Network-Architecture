class UltraSMHT(nn.Module):
    def __init__(self, a=2.0, b=1.2, c=3.0, d=0.8, g=2.0, f=1.1, h=2.0, i=0.9):
        super().__init__()
        param_names = ['a', 'b', 'c', 'd', 'g', 'f', 'h', 'i']
        for name, value in zip(param_names, [a, b, c, d, g, f, h, i]):
         setattr(self, name, nn.Parameter(torch.tensor(value)))

    def forward(self, x):
        base = lambda p: torch.clamp(p, min=1e-3)
        num = torch.pow(base(self.a), self.b * x) - torch.pow(base(self.c), self.d * x)
        denom = torch.pow(base(self.g), self.f * x) + torch.pow(base(self.h), self.i * x) + 1e-8
        return num / denom

# === MLP using UltraSMHT ===
class DeepMLPWithUltraSMHT(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.act = UltraSMHT()
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        x = self.output_layer(x)
        return x
