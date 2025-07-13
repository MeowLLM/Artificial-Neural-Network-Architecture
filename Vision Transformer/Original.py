import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. GatedMLP Class (as provided by the user) ---
class GatedMLP(nn.Module):
    """
    Implements a Gated Multi-Layer Perceptron.
    This MLP incorporates a per-node gating mechanism to control information flow.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initializes the GatedMLP.

        Args:
            input_dim (int): The dimension of the input features.
            hidden_dim (int): The dimension of the hidden layer.
            output_dim (int): The dimension of the output features.
        """
        super(GatedMLP, self).__init__()
        # First linear layer to project input to hidden dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Gate layer: learns a gate value for each node in the hidden layer
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)
        # Second linear layer to project gated hidden representation to output dimension
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GatedMLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - out (torch.Tensor): The final output of the GatedMLP.
                - gate_val (torch.Tensor): The computed gate values.
        """
        # Pass input through the first linear layer and apply ReLU activation
        h1 = F.relu(self.fc1(x))
        # Compute gate values using the gate_layer and sigmoid activation
        # Sigmoid ensures gate values are between 0 and 1
        gate_val = torch.sigmoid(self.gate_layer(h1))
        # Apply gating: element-wise multiplication of h1 by gate_val
        # This selectively scales activations based on the learned gate
        h1_gated = h1 * gate_val
        # Pass the gated hidden representation through the second linear layer
        out = self.fc2(h1_gated)
        return out, gate_val

# --- 2. Patch Embedding Layer ---
class PatchEmbedding(nn.Module):
    """
    Transforms an input image into a sequence of flattened, linearly projected
    patches, with an added learnable CLS token and positional embeddings.
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """
        Initializes the PatchEmbedding layer.

        Args:
            img_size (int): The size of the input image (assuming square images).
            patch_size (int): The size of each square patch.
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): The dimension to which each patch embedding is projected.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Convolutional layer to extract patches and project them to embed_dim
        # kernel_size and stride equal to patch_size ensure non-overlapping patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings for each patch + CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbedding layer.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Embedded patches with CLS token and positional embeddings,
                          of shape (B, num_patches + 1, embed_dim).
        """
        B, C, H, W = x.shape
        # Ensure image dimensions are divisible by patch size
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})."

        # Project image into patches: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        # Then flatten the spatial dimensions: (B, embed_dim, N_h, N_w) -> (B, embed_dim, N_patches)
        # Finally, transpose to (B, N_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)

        # Expand CLS token to match batch size
        cls_token = self.cls_token.expand(B, -1, -1)
        # Concatenate CLS token with patch embeddings
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        # Apply dropout
        x = self.dropout(x)
        return x

# --- 3. Multi-Head Self-Attention ---
class MultiHeadSelfAttention(nn.Module):
    """
    Implements a Multi-Head Self-Attention module.
    Uses PyTorch's built-in MultiheadAttention for efficiency.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Initializes the MultiHeadSelfAttention module.

        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout_rate (float): Dropout rate for attention weights.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        # PyTorch's MultiheadAttention expects (sequence_length, batch_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiHeadSelfAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, E), where S is sequence length, E is embed_dim.

        Returns:
            torch.Tensor: Output tensor after self-attention and residual connection,
                          of shape (B, S, E).
        """
        # Apply Layer Normalization before attention
        norm_x = self.norm(x)
        # Apply multi-head self-attention
        # attn_output: (B, S, E), attn_output_weights: (B, S, S)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        # Apply dropout
        attn_output = self.dropout(attn_output)
        # Add residual connection
        return x + attn_output

# --- 4. MLP Block (Feed-Forward Network) ---
class MLPBlock(nn.Module):
    """
    Implements a standard MLP block with Layer Normalization, two linear layers,
    GELU activation, and dropout, followed by a residual connection.
    """
    def __init__(self, embed_dim: int, mlp_dim: int, dropout_rate: float = 0.1):
        """
        Initializes the MLPBlock.

        Args:
            embed_dim (int): The dimension of the input and output embeddings.
            mlp_dim (int): The dimension of the hidden layer in the MLP.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, E).

        Returns:
            torch.Tensor: Output tensor after MLP and residual connection,
                          of shape (B, S, E).
        """
        # Apply Layer Normalization before MLP
        norm_x = self.norm(x)
        # Apply MLP
        mlp_output = self.mlp(norm_x)
        # Add residual connection
        return x + mlp_output

# --- 5. Transformer Block ---
class TransformerBlock(nn.Module):
    """
    Represents a single Transformer Encoder Block, combining Multi-Head Self-Attention
    and an MLP Block, each with Layer Normalization and residual connections.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout_rate: float = 0.1):
        """
        Initializes the TransformerBlock.

        Args:
            embed_dim (int): The dimension of the embeddings.
            num_heads (int): Number of attention heads.
            mlp_dim (int): Hidden dimension for the MLP block.
            dropout_rate (float): Dropout rate for attention and MLP.
        """
        super().__init__()
        # Attention sub-block
        self.attention_block = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        # MLP sub-block
        self.mlp_block = MLPBlock(embed_dim, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, E).

        Returns:
            torch.Tensor: Output tensor after passing through attention and MLP,
                          of shape (B, S, E).
        """
        # Pass through attention block (includes LayerNorm and residual)
        x = self.attention_block(x)
        # Pass through MLP block (includes LayerNorm and residual)
        x = self.mlp_block(x)
        return x

# --- 6. EfficientVisionTransformer Model ---
class EfficientVisionTransformer(nn.Module):
    """
    Implements the Efficient Vision Transformer model based on the provided diagram.
    It processes images through patch embedding, a sequence of transformer blocks,
    and then uses a GatedMLP for final classification.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, num_transformer_blocks: int = 12,
                 num_heads: int = 12, mlp_dim_ratio: int = 4, num_classes: int = 100,
                 dropout_rate: float = 0.1):
        """
        Initializes the EfficientVisionTransformer.

        Args:
            img_size (int): Size of the input image (e.g., 224 for 224x224).
            patch_size (int): Size of image patches (e.g., 16 for 16x16 patches).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of patch embeddings and transformer hidden states.
            num_transformer_blocks (int): Number of Transformer Encoder Blocks.
            num_heads (int): Number of attention heads in MultiHeadSelfAttention.
            mlp_dim_ratio (int): Ratio to determine MLP hidden dimension (mlp_dim = embed_dim * mlp_dim_ratio).
            num_classes (int): Number of output classes for classification.
            dropout_rate (float): Dropout rate used throughout the model.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.mlp_dim = embed_dim * mlp_dim_ratio

        # 1. Patch Embedding Layer
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # 2. Transformer Blocks (Encoder)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=dropout_rate
            ) for _ in range(num_transformer_blocks)
        ])

        # 3. Final Layer Normalization before classification head
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Classification Head: GatedMLP and a standard MLP in parallel
        # The diagram shows both GateMLP and MLP leading to the output after Majority Voting.
        # For simplicity, we'll use the GatedMLP as the primary classification head,
        # and also include a standard MLP for comparison or potential ensemble.
        # We'll assume the CLS token output is fed into these.
        self.gated_mlp_head = GatedMLP(
            input_dim=embed_dim,
            hidden_dim=self.mlp_dim, # Can be adjusted
            output_dim=num_classes
        )
        self.standard_mlp_head = nn.Sequential(
            nn.Linear(embed_dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.mlp_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EfficientVisionTransformer.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits for classification, of shape (B, num_classes).
        """
        # 1. Patch Embedding
        # Output shape: (B, num_patches + 1, embed_dim)
        x = self.patch_embedding(x)

        # 2. Pass through Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # 3. Extract CLS token for classification
        # The CLS token is at index 0 of the sequence dimension
        cls_token_output = x[:, 0]

        # 4. Apply final Layer Normalization
        cls_token_output = self.norm(cls_token_output)

        # 5. Classification with GatedMLP head
        # The diagram implies these are final classification layers.
        # We return the output from the GatedMLP head.
        # You could combine outputs from both gated_mlp_head and standard_mlp_head
        # if you want an ensemble or different final layer architecture.
        logits, _ = self.gated_mlp_head(cls_token_output)

        # If you wanted to use the standard MLP head:
        # logits = self.standard_mlp_head(cls_token_output)

        # If you wanted to combine them (e.g., average their outputs):
        # logits_gated, _ = self.gated_mlp_head(cls_token_output)
        # logits_standard = self.standard_mlp_head(cls_token_output)
        # logits = (logits_gated + logits_standard) / 2 # Simple average

        return logits

# --- Example Usage ---
if __name__ == "__main__":
    # Define model parameters
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 512
    num_transformer_blocks = 12
    num_heads = 8
    mlp_dim_ratio = 4 # MLP hidden dim will be embed_dim * 4
    num_classes = 100
    dropout_rate = 0.1

    print("--- Testing GatedMLP ---")
    gmlp = GatedMLP(input_dim=128, hidden_dim=256, output_dim=10)
    dummy_input_gmlp = torch.randn(4, 128) # Batch size 4, input dim 128
    output_gmlp, gate_vals = gmlp(dummy_input_gmlp)
    print(f"GatedMLP Output Shape: {output_gmlp.shape}")
    print(f"Gate Values Shape: {gate_vals.shape}")
    print("-" * 30)

    print("--- Initializing EfficientVisionTransformer ---")
    model = EfficientVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        mlp_dim_ratio=mlp_dim_ratio,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    print(f"Model successfully initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print("-" * 30)

    # Create a dummy input image (batch_size, channels, height, width)
    dummy_image = torch.randn(1, in_channels, img_size, img_size)
    print(f"Dummy Input Image Shape: {dummy_image.shape}")

    # Perform a forward pass
    print("--- Performing forward pass ---")
    try:
        output_logits = model(dummy_image)
        print(f"Output Logits Shape: {output_logits.shape}")
        print(f"Output Logits (first 5 values): {output_logits[0, :5]}")
    except Exception as e:
        print(f"An error occurred during forward pass: {e}")

    print("\n--- Model Architecture Summary ---")
    # You can uncomment the following lines to print a detailed summary
    # from torchinfo import summary
    # summary(model, input_size=(1, in_channels, img_size, img_size))
