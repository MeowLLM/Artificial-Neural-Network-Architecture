class PatchEmbedding(nn.Module):
    """
    Transforms input images into a sequence of flattened patches and projects them
    into a higher-dimensional embedding space.
    """
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        # Calculate the number of patches along one side, then square for total patches
        self.num_patches = (img_size // patch_size) ** 2
        # Use a convolutional layer with kernel_size and stride equal to patch_size
        # to effectively extract and project patches.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H_new, W_new)
        x = self.proj(x)
        # Flatten the spatial dimensions (H_new * W_new) into a single dimension
        # and then transpose to (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the patch embeddings.
    This helps the model understand the spatial relationships between patches.
    """
    def __init__(self, embed_dim: int, num_patches: int):
        super().__init__()
        # Learnable positional embeddings. We add +1 for a learnable [CLS] token.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create a dummy [CLS] token for each item in the batch.
        # This token typically aggregates global information in ViTs.
        cls_token = torch.randn(x.shape[0], 1, x.shape[2], device=x.device)
        # Concatenate the [CLS] token with the patch embeddings along the sequence dimension
        x = torch.cat((cls_token, x), dim=1)
        # Add the positional embeddings to the combined sequence
        x = x + self.pos_embedding
        return x

# --- ViTBlockOuter Class (The optimized ViT backbone) ---
class ViTBlockOuter(nn.Module):
    """
    Represents the Vision Transformer backbone, including patch embedding,
    positional encoding, transformer encoder layers, adaptive pooling,
    and the final projection layer.
    """
    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 128, # Core embedding dimension for the ViT backbone
                 num_layers: int = 6, # Number of Transformer Encoder layers
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 output_projection_dim: int = 2048): # Target output size after ViT path projection
        super().__init__()

        # Patch Embedding layer
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        # Positional Encoding layer
        self.positional_encoding = PositionalEncoding(embed_dim, self.patch_embedding.num_patches)

        # Transformer Encoder stack
        # `nn.TransformerEncoderLayer` defines a single encoder block (Multi-Head Attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,         # Input and output feature dimension
            nhead=num_heads,           # Number of attention heads
            dim_feedforward=embed_dim * mlp_ratio, # Hidden dimension of the FFN
            batch_first=True           # Input/output tensors are (batch, sequence, feature)
        )
        # `nn.TransformerEncoder` stacks multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection layer for the ViT path output
        # It takes the pooled `embed_dim` output and projects it to `output_projection_dim`
        self.vit_output_proj = nn.Linear(embed_dim, output_projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Apply Patch Embedding
        # Output: (B, num_patches, embed_dim)
        x = self.patch_embedding(x)

        # 2. Add Positional Encoding (with a dummy CLS token)
        # Output: (B, num_patches + 1, embed_dim)
        x = self.positional_encoding(x)

        # 3. Pass through Transformer Encoder layers
        # Output: (B, num_patches + 1, embed_dim)
        vit_path_output = self.transformer_encoder(x)

        # 4. Apply Adaptive Pooling (mean pooling over the sequence dimension)
        # This reduces the sequence of tokens to a single feature vector per batch item.
        # (B, embed_dim)
        vit_path_output_pooled = vit_path_output.mean(dim=1)

        # 5. Project the pooled output to the target dimension
        # (B, output_projection_dim)
        vit_path_output_projected = self.vit_output_proj(vit_path_output_pooled)

        return vit_path_output_projected

# --- ConvAlphaBlock Class (The convolutional path) ---
class ConvAlphaBlock(nn.Module):
    """
    A simplified convolutional block that processes images and projects its
    flattened output to a specified latent dimension.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_projection_dim: int = 2048, # Target output size after Conv path projection
                 img_size: int = 224): # Input image size for calculating flattened dimension
        super().__init__()
        # Define a sequence of convolutional layers with ReLU activation and Max Pooling
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces spatial dimensions by half
            nn.Conv2d(out_channels, out_channels * 2, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(2) # Reduces spatial dimensions by half again
        )

        # Calculate the flattened dimension after the convolutional and pooling layers
        # Assuming square images and 'same' padding for simplicity
        # After first MaxPool: img_size / 2
        # After second MaxPool: (img_size / 2) / 2 = img_size / 4
        final_spatial_dim = img_size // 4
        # The number of channels after the last conv layer is `out_channels * 2`
        conv_intermediate_flat_dim = (out_channels * 2) * final_spatial_dim * final_spatial_dim

        # Projection layer for the convolutional path output
        # It takes the flattened convolutional features and projects them to `output_projection_dim`
        self.conv_output_proj = nn.Linear(conv_intermediate_flat_dim, output_projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through convolutional layers
        x = self.conv_layers(x)
        # Flatten the output of the convolutional layers into a 1D vector
        conv_alpha_output_flat = x.flatten(1)
        # Project the flattened output to the target dimension
        conv_alpha_output_projected = self.conv_output_proj(conv_alpha_output_flat)
        return conv_alpha_output_projected

# --- HybridModel Class (Combines ViT and Conv paths) ---
class HybridModel(nn.Module):
    """
    A conceptual hybrid model that combines a Vision Transformer path
    and a Convolutional Neural Network path, then fuses their outputs.
    """
    def __init__(self,
                 vit_img_size: int = 224,
                 vit_in_channels: int = 3,
                 vit_patch_size: int = 16,
                 vit_embed_dim: int = 128, # ViT backbone embedding dimension
                 vit_num_layers: int = 6,
                 vit_num_heads: int = 8,
                 vit_mlp_ratio: int = 4,
                 common_projection_dim: int = 2048, # Common dimension for both paths' outputs
                 conv_in_channels: int = 3,
                 conv_out_channels: int = 32,
                 conv_kernel_size: int = 3,
                 conv_stride: int = 1,
                 conv_padding: int = 1):
        super().__init__()

        # Instantiate the Vision Transformer path
        self.vit_path = ViTBlockOuter(
            img_size=vit_img_size,
            in_channels=vit_in_channels,
            patch_size=vit_patch_size,
            embed_dim=vit_embed_dim,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            output_projection_dim=common_projection_dim # ViT projects to the common dimension
        )

        # Instantiate the Convolutional Alpha Block path
        self.conv_path = ConvAlphaBlock(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            output_projection_dim=common_projection_dim, # Conv projects to the common dimension
            img_size=vit_img_size # Assuming same input image size for both paths
        )

        # Example of a fusion layer (e.g., a classification head)
        # This layer takes the combined output from both paths
        self.fusion_layer = nn.Linear(common_projection_dim, 10) # Example: 10 output classes

    def forward(self, x_vit: torch.Tensor, x_conv: torch.Tensor) -> torch.Tensor:
        # Process input through the ViT path
        vit_output = self.vit_path(x_vit)
        # Process input through the Conv path
        conv_output = self.conv_path(x_conv)

        # Combine the outputs from both paths.
        # Here, we sum them, assuming they represent complementary features.
        # Other fusion strategies could be concatenation, element-wise product, etc.
        combined_output = vit_output + conv_output

        # Pass the combined output through the final fusion/classification layer
        output = self.fusion_layer(combined_output)
        return output
