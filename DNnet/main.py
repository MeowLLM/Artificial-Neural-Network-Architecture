import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== DNConv Expert =====
# This class represents a basic convolutional expert module.
# In the diagram, this corresponds to the 'DNConv N', 'DNConv 2', 'DNConv 1'
# modules within the 'DN-Block'. It's a fundamental building block.
class DNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A standard 2D convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # Batch Normalization for stable training
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply convolution, batch normalization, and ReLU sequentially
        return self.relu(self.bn(self.conv(x)))

# ===== MoE Selector for Expert Paths =====
# This class implements the Mixture of Experts (MoE) gating mechanism.
# It corresponds to the 'MoE: Select output' block in the 'DN-Block' diagram.
# It learns to assign weights to each expert based on the input features.
class MoESelect(nn.Module):
    def __init__(self, num_experts, in_channels):
        super().__init__()
        # The gating network determines which expert to use.
        # It takes the input features and outputs a probability distribution
        # over the experts using a Softmax layer.
        self.gate = nn.Sequential(
            # Global average pooling to reduce spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d(1),
            # Flatten the output to a 1D vector
            nn.Flatten(),
            # Linear layer to project features to the number of experts
            nn.Linear(in_channels, num_experts),
            # Softmax to get a probability distribution over experts
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Pass the input through the gating network to get expert weights
        return self.gate(x)

# ===== DNBlock with Memory, Gating, MoE, and Recursive Thinking =====
# This is the core block of the architecture, corresponding to the 'DN-Block' diagram.
# It integrates an initial convolution, a Mixture of Experts (MoE),
# a recursive processing loop, and an early exit gate.
class DNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, max_recursions=32):
        super().__init__()
        self.max_recursions = max_recursions # Maximum number of recursive steps

        # Initial convolution layer, similar to the 'Conv' at the input of 'DN-Block'
        self.init_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # ModuleList to hold multiple DNConv experts.
        # These are 'DNConv N', 'DNConv 2', 'DNConv 1' in the diagram.
        self.experts = nn.ModuleList([
            DNConv(out_channels, out_channels) for _ in range(num_experts)
        ])
        # The MoE selector to dynamically choose/weight experts.
        self.moe = MoESelect(num_experts, out_channels)

        # An 'exit gate' to decide when to stop the recursion early.
        # This is not explicitly labeled in the 'DN-Block' diagram but
        # represents a dynamic exit mechanism, similar to what 'Output Hidden'
        # and conditional paths might imply.
        self.exit_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global average pooling
            nn.Flatten(),            # Flatten
            nn.Linear(out_channels, 1), # Single output neuron
            nn.Sigmoid()             # Sigmoid to get a probability (0 to 1)
        )

    def forward(self, x):
        # Process the initial input, creating an 'original cache' or initial state.
        # This is the first step after 'Input' and 'Conv' in the 'DN-Block'.
        cache_original = self.relu(self.init_conv(x))
        # 'cache_node' represents the current state that gets recursively processed.
        # It's initialized with a clone of the original cache.
        cache_node = cache_original.clone()

        # Recursive thinking loop: The block can process the input multiple times.
        # This loop represents the internal feedback implied by 'Hidden Output'
        # and the iterative nature of the 'DN-Block'.
        for recursion_step in range(self.max_recursions):
            # Get weights for each expert based on the current 'cache_node' state.
            # The .view() reshapes weights for element-wise multiplication with expert outputs.
            weights = self.moe(cache_node).view(cache_node.size(0), -1, 1, 1, 1)

            # Apply each expert to the current 'cache_node' and stack their outputs.
            outputs = torch.stack([expert(cache_node) for expert in self.experts], dim=1)

            # Combine expert outputs using the learned weights.
            # This is the weighted sum of expert outputs.
            combined = (outputs * weights).sum(dim=1)

            # Calculate the probability of exiting the recursion.
            exit_prob = self.exit_gate(combined).squeeze(1)

            # If the exit probability for all samples in the batch is above a threshold (0.5),
            # we exit early and return the combined output. This is the dynamic exit.
            if (exit_prob > 0.5).all():
                # print(f"Exiting DNBlock early at recursion step {recursion_step + 1}") # For debugging
                return combined

            # If not exiting, update 'cache_node' for the next recursive step.
            # This implements the feedback loop within the 'DN-Block'.
            cache_node = combined

        # If the loop completes without early exit, return the final 'cache_node'.
        return cache_node

# ===== Full Architecture (Stacked DNBlocks + Residual Logic) =====
# This class represents the entire DNNet, stacking multiple DNBlocks.
# It corresponds to the 'Example connection' diagram, showing stacked DN-Blocks
# and a final residual connection.
class DNNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=4, num_experts=3, max_recursions=32):
        super().__init__()
        # Initial encoder convolution to map input channels to base_channels.
        self.encoder = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Stack multiple DNBlocks. Each block learns to process the features.
        self.blocks = nn.ModuleList([
            DNBlock(base_channels, base_channels, num_experts, max_recursions)
            for _ in range(num_blocks)
        ])

        # Final convolution to map features back to the original input channel size.
        self.final_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x_input = x # Store the original input for the residual connection.

        # Pass through the initial encoder.
        x = self.encoder(x)

        # Pass through each DNBlock sequentially.
        for block in self.blocks:
            x = block(x)

        # Apply the final convolution.
        x = self.final_conv(x)

        # Residual output: Subtract the learned 'noise' or 'transformation' (x)
        # from the original input (x_input). This is common in denoising networks
        # like DnCNN, and aligns with 'Output residual image' in the diagram.
        return x_input - x
