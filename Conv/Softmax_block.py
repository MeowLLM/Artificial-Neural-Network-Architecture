class SoftmaxBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temperature1=1.0, temperature2=1.0):
        super(SoftmaxBlock, self).__init__()
        self.temp1 = temperature1
        self.temp2 = temperature2

        # Main branch
        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Concat branch
        self.concat_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        # Gate branches
        self.gate_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gate_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gate_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # ---------------- Main branch ----------------
        main = self.relu(self.main_conv(x))  # Conv -> ReLU

        # ---------------- Concat branch ----------------
        c1 = self.relu(self.concat_conv1(x))         # Conv -> ReLU
        concat_out = torch.cat([main, c1], dim=1)    # concat channels
        concat_out = self.relu(self.concat_conv2(concat_out))  # Concat -> ReLU

        # ---------------- Gate branches ----------------
        g1 = self.relu(self.gate_conv1(x))  # Conv -> ReLU
        g2 = self.relu(self.gate_conv2(x))  # Conv -> ReLU
        g3 = self.relu(self.gate_conv3(x))  # Conv -> ReLU

        # SoftmaxTemp2 processing
        gate_merge = g1 + g2 + g3
        gate_out = F.softmax(gate_merge / self.temp2, dim=1)

        # ---------------- Merge all to SoftmaxTemp1 ----------------
        merged = main + concat_out + gate_out
        out = F.softmax(merged / self.temp1, dim=1)

        return out
