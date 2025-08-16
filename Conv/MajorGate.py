class ConfidenceGate(nn.Module):
    """
    A PyTorch implementation of the ConfidenceGate architecture described in the diagram.

    This module consists of four parallel convolutional branches, each representing
    a different scenario of confidence and accuracy. A Gumbel-Softmax layer is used
    to select or combine the outputs of these branches.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the ConfidenceGate module.

        Args:
            in_channels (int): The number of input channels for the convolutional layers.
            out_channels (int): The number of output channels for the convolutional layers.
        """
        super(ConfidenceGate, self).__init__()

        # --- Branch 1: Low confidence + High accuracy ---
        self.branch1_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # --- Branch 2: High confidence + Low accuracy ---
        self.branch2_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # --- Branch 3: High confidence + High accuracy ---
        self.branch3_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # --- Branch 4: Low confidence + Low accuracy ---
        self.branch4_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # This layer will be used to generate the selection logits for the Gumbel-Softmax.
        # It takes the concatenated gate inputs and outputs 4 values, one for each branch.
        self.gate_selector = nn.Linear(in_channels * 2, 4) # Assuming gate inputs are concatenated

    def forward(self, x, gate_input1, gate_input2, confidence, accuracy):
        """
        Defines the forward pass of the ConfidenceGate with updated connection logic.

        Args:
            x (torch.Tensor): The main input tensor.
            gate_input1 (torch.Tensor): The first gate input tensor.
            gate_input2 (torch.Tensor): The second gate input tensor.
            confidence (torch.Tensor): A tensor representing the confidence score.
            accuracy (torch.Tensor): A tensor representing the accuracy score.

        Returns:
            torch.Tensor: The final output of the module.
        """
        # --- Process each branch based on the new connection rules ---

        # Branch 1: GateIn --> Conv --> ReLU
        b1_out = F.relu(self.branch1_conv(gate_input1))

        # Branch 2: x --> Conv --> ReLU
        b2_out = F.relu(self.branch2_conv(x))

        # Branch 3: GateIn --> Conv --> ReLU
        b3_out = F.relu(self.branch3_conv(gate_input2))

        # Branch 4: x --> Conv --> ReLU --> Punish by confidence and accuracy
        b4_conv_out = F.relu(self.branch4_conv(x))
        # The "punishment" is interpreted as scaling the output down.
        # We assume confidence and accuracy are values between 0 and 1.
        # The punishment factor will be broadcastable to the tensor's shape.
        punishment_factor = (1.0 - confidence) * (1.0 - accuracy)
        b4_out = b4_conv_out * punishment_factor


        # --- Output Selection using Gumbel-Softmax ---

        # The selection logic remains the same, using a combination of the main
        # input and a gate input to decide which branch to activate.
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        gate_input1_flat = F.adaptive_avg_pool2d(gate_input1, (1, 1)).view(gate_input1.size(0), -1)
        
        gate_features = torch.cat((avg_pool, gate_input1_flat), dim=1)
        
        # Get the logits for each of the 4 branches.
        gate_logits = self.gate_selector(gate_features)

        # Apply Gumbel-Softmax to get a differentiable, categorical-like distribution.
        # The `hard=True` argument makes the output one-hot in the forward pass.
        gate_weights = F.gumbel_softmax(gate_logits, tau=1, hard=True)

        # Stack the branch outputs to easily select one.
        # [N, C, H, W] -> [N, 1, C, H, W]
        b1_out_s = b1_out.unsqueeze(1)
        b2_out_s = b2_out.unsqueeze(1)
        b3_out_s = b3_out.unsqueeze(1)
        b4_out_s = b4_out.unsqueeze(1)

        # [N, 4, C, H, W]
        all_branches = torch.cat([b1_out_s, b2_out_s, b3_out_s, b4_out_s], dim=1)

        # Use the gate_weights to select the output.
        # gate_weights shape: [N, 4]
        # We need to reshape it to [N, 4, 1, 1, 1] to multiply with the branches tensor.
        gate_weights_reshaped = gate_weights.view(gate_weights.size(0), 4, 1, 1, 1)

        # Multiply the weights with the branch outputs and sum them up.
        # Since the weights are one-hot, this effectively selects one branch.
        selected_output = (all_branches * gate_weights_reshaped).sum(dim=1)

        return selected_output
