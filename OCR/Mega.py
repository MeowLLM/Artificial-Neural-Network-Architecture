import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Core Building Blocks ---

class ConvBlock(nn.Module):
    """
    A standard Convolutional Block: Conv2d -> BatchNorm2d -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class StrideConv2dBlock(nn.Module):
    """
    Strided Convolutional Block: Conv2d (stride > 1) -> BatchNorm2d -> ReLU -> Dropout.
    Used for downsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dropout_rate=0.2):
        super(StrideConv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) # Using Dropout2d for spatial dropout

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class DeConv2dBlock(nn.Module):
    """
    Deconvolutional (ConvTranspose2d) Block: ConvTranspose2d -> BatchNorm2d -> ReLU.
    Used for upsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(DeConv2dBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron Block: Linear -> ReLU -> Dropout -> Linear.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(MLPBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class BiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM Block.
    Expects input of shape (batch_size, sequence_length, input_size).
    Outputs (batch_size, sequence_length, hidden_size * 2).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.2):
        super(BiLSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

    def forward(self, x):
        # h_0 and c_0 are initialized to zeros by default if not provided
        output, _ = self.lstm(x)
        return output

class MajorityVoting(nn.Module):
    """
    A simple Majority Voting mechanism, implemented as averaging for continuous outputs.
    For classification, this would typically involve taking the mode of predictions.
    Here, it averages the input along a specified dimension.
    """
    def __init__(self, dim=-1):
        super(MajorityVoting, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)

# --- 2. Gating Mechanisms ---

class GateSelect(nn.Module):
    """
    A learned gating mechanism to blend between two input paths.
    It learns a weight 'alpha' (0 to 1) for the first input, and (1-alpha) for the second.
    Assumes inputs are of compatible shapes for element-wise operations.
    """
    def __init__(self, input_dim):
        super(GateSelect, self).__init__()
        # A simple linear layer to learn the gating scalar
        # The input_dim here should correspond to a feature dimension if the gate is feature-wise,
        # or just 1 if it's a global gate. For simplicity, let's make it a global gate.
        self.gate_weight = nn.Parameter(torch.tensor(0.5)) # Initialize to blend equally

    def forward(self, x1, x2):
        # Ensure gate_weight is between 0 and 1 using sigmoid
        alpha = torch.sigmoid(self.gate_weight)
        return alpha * x1 + (1 - alpha) * x2

class GateDenoise(nn.Module):
    """
    A spatial gating mechanism for denoising, as specified by the user.
    Generates a spatial gate map based on global features.
    """
    def __init__(self, in_channels):
        super(GateDenoise, self).__init__()
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # Global feature (fast!) -> (B, C, 1, 1)
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1), # Squeeze -> (B, C/2, 1, 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1), # Excite -> (B, 1, 1, 1)
            nn.Sigmoid()                  # Outputs gate ∈ [0, 1]
        )

    def forward(self, x):
        # x: (B, C, H, W)
        gate = self.gate_fc(x) # (B, 1, 1, 1) - this will broadcast across C, H, W
        return x * gate # Apply spatial attention (broadcasted)

class GatedMLP(nn.Module): # Renamed from GateMLP to GatedMLP as per user's request
    """
    Multi-Layer Perceptron with per-node gating, as specified by the user.
    """
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
        return out, gate_val # Returns both output and gate_val

# --- 3. Higher-Level Blocks ---

class EncoderDecoderPath(nn.Module): # Renamed from DenoiseBlock for clarity
    """
    Encoder-Decoder Path: Strided Conv2d -> DeConv2d.
    This is the core denoising path, separated from the gating mechanism.
    """
    def __init__(self, in_channels, hidden_channels):
        super(EncoderDecoderPath, self).__init__()
        self.encoder = StrideConv2dBlock(in_channels, hidden_channels)
        # To ensure output_padding works, calculate the output shape of encoder
        # and then determine deconv parameters. For simplicity, assume symmetric.
        self.decoder = DeConv2dBlock(hidden_channels, in_channels) # Output same channels as input

    def forward(self, x):
        return self.decoder(self.encoder(x))

class ConvAlphaBlock(nn.Module):
    """
    Conv-Alpha Block: GateDenoise -> EncoderDecoderPath -> ConvBlock -> Flattening.
    """
    def __init__(self, in_channels, hidden_denoise_channels, conv_out_channels, img_height, img_width):
        super(ConvAlphaBlock, self).__init__()
        self.gate_denoise = GateDenoise(in_channels) # New GateDenoise takes in_channels
        self.encoder_decoder_path = EncoderDecoderPath(in_channels, hidden_denoise_channels) # Renamed
        self.conv_block = ConvBlock(in_channels, conv_out_channels) # Output of encoder_decoder_path is in_channels
        
        # Calculate flattened size:
        # Assuming EncoderDecoderPath maintains spatial dimensions or roughly so,
        # and ConvBlock maintains spatial dimensions with padding=1, kernel_size=3, stride=1.
        self.flattened_size = conv_out_channels * img_height * img_width # Assuming no spatial change

    def forward(self, x):
        gated_input = self.gate_denoise(x) # Apply new GateDenoise
        denoised_output = self.encoder_decoder_path(gated_input) # Pass through encoder-decoder
        conv_output = self.conv_block(denoised_output)
        flattened_output = torch.flatten(conv_output, 1) # Flatten starting from batch dimension

        return flattened_output

class PatchEmbedding(nn.Module):
    """
    Splits image into patches, flattens them, and applies a linear projection.
    Adds learnable positional embeddings.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim, dropout_rate=0.1):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x) # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, embed_dim)
        x = x + self.pos_embedding # Add positional embedding
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """
    A standard Transformer Encoder Block.
    Consists of Multi-Head Self-Attention, Layer Normalization, and Feed-Forward Network.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(), # Common activation in Transformers
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Attention block
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output # Residual connection

        # MLP block
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output # Residual connection
        return x

class ViTInnerBlock(nn.Module):
    """
    Represents the inner 'ViT-Block' from the diagram, which is a sequence of TransformerBlocks.
    """
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4., dropout_rate=0.1):
        super(ViTInnerBlock, self).__init__()
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        return x

# --- 4. Composite Blocks from Diagram ---

class ViTBlockOuter(nn.Module):
    """
    The main ViT-Block from the top-left diagram.
    Input -> Gate Select -> (ViT-Block (inner) / Conv-Alpha Block) -> Process Block -> Output.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_transformer_layers,
                 num_heads, mlp_ratio, hidden_denoise_channels, conv_out_channels, process_block_out_dim):
        super(ViTBlockOuter, self).__init__()
        
        # Patch Embedding for ViT path
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Inner ViT Block (sequence of Transformer Blocks)
        self.vit_inner_block = ViTInnerBlock(embed_dim, num_heads, num_transformer_layers, mlp_ratio)
        
        # Conv-Alpha Block
        self.conv_alpha_block = ConvAlphaBlock(in_channels, hidden_denoise_channels, conv_out_channels, img_size, img_size)
        
        # Gate Select to choose between ViT path and Conv-Alpha path
        # The output dimensions of the two paths need to be compatible for GateSelect.
        # ViT path output: (batch_size, num_patches, embed_dim)
        # Conv-Alpha path output: (batch_size, flattened_size)
        # This requires a projection layer before GateSelect to match dimensions.
        
        # Project ViT output to match flattened_size from Conv-Alpha
        # Corrected: Use self.patch_embed.num_patches for the input dimension
        self.vit_output_proj = nn.Linear(self.patch_embed.num_patches * embed_dim, self.conv_alpha_block.flattened_size)
        # Note: This projection is a simplification. A more robust solution might involve
        # an adaptive pooling or a more complex fusion. For now, flatten and project.

        self.gate_select = GateSelect(1) # Global gate

        # Process Block: A simple MLP for demonstration
        self.process_block = MLPBlock(self.conv_alpha_block.flattened_size, 
                                      self.conv_alpha_block.flattened_size // 2, 
                                      process_block_out_dim)

    def forward(self, x):
        # Path 1: ViT Block
        vit_path_output = self.patch_embed(x) # (B, num_patches, embed_dim)
        vit_path_output = self.vit_inner_block(vit_path_output) # (B, num_patches, embed_dim)
        
        # Flatten and project ViT output to match Conv-Alpha output for GateSelect
        vit_path_output_flat = vit_path_output.flatten(1) # (B, num_patches * embed_dim)
        vit_path_output_projected = self.vit_output_proj(vit_path_output_flat) # (B, flattened_size)

        # Path 2: Conv-Alpha Block
        conv_alpha_output = self.conv_alpha_block(x) # (B, flattened_size)

        # Apply Gate Select
        gated_output = self.gate_select(vit_path_output_projected, conv_alpha_output)

        # Process Block
        output = self.process_block(gated_output)
        return output

class CombinedMLPBiLSTMBlock(nn.Module):
    """
    Block combining MLP and BiLSTM with a GatedMLP and Majority Voting.
    Input -> (GatedMLP / BiLSTM) -> GateMLP (blending) -> Majority Voting -> Connectionist Temporal Classification (CTC).
    CTC is a loss function, not a layer, so it's applied externally.
    """
    def __init__(self, input_dim_mlp, hidden_dim_mlp, output_dim_mlp,
                 input_size_bilstm, hidden_size_bilstm,
                 sequence_length_for_bilstm_input, # Needs to be known for MLP projection
                 majority_voting_dim=-1):
        super(CombinedMLPBiLSTMBlock, self).__init__()

        self.mlp_block = GatedMLP(input_dim_mlp, hidden_dim_mlp, output_dim_mlp) # Using GatedMLP
        
        self.bilstm_block = BiLSTMBlock(input_size_bilstm, hidden_size_bilstm)

        self.bilstm_proj_dim = hidden_size_bilstm * 2 * sequence_length_for_bilstm_input
        self.bilstm_output_proj = nn.Linear(self.bilstm_proj_dim, output_dim_mlp) # Project BiLSTM output to MLP output dim

        self.gate_mlp = GateSelect(1) # Still using GateSelect for blending MLP and BiLSTM outputs
        self.majority_voting = MajorityVoting(dim=majority_voting_dim) # Apply after gating

    def forward(self, x_mlp, x_bilstm):
        # x_mlp: (batch_size, input_dim_mlp)
        # x_bilstm: (batch_size, sequence_length, input_size_bilstm)

        mlp_output, _ = self.mlp_block(x_mlp) # Get only the output from GatedMLP # (batch_size, output_dim_mlp)

        bilstm_output = self.bilstm_block(x_bilstm) # (batch_size, sequence_length, hidden_size * 2)
        
        # Flatten BiLSTM output and project to match MLP output dimension
        bilstm_output_flat = bilstm_output.flatten(1) # (batch_size, sequence_length * hidden_size * 2)
        bilstm_output_projected = self.bilstm_output_proj(bilstm_output_flat) # (batch_size, output_dim_mlp)

        # Gate between MLP and BiLSTM outputs
        gated_output = self.gate_mlp(mlp_output, bilstm_output_projected)

        # Apply Majority Voting (e.g., mean across features if dim=-1)
        final_output = self.majority_voting(gated_output)
        return final_output
