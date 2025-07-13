import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. DeepGatedMLP Class (updated from GatedMLP) ---
class DeepGatedMLP(nn.Module):
    """
    Implements a Deep Gated Multi-Layer Perceptron.
    This MLP incorporates per-node gating mechanisms across multiple hidden layers
    to control information flow and increase depth.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """
        Initializes the DeepGatedMLP.

        Args:
            input_dim (int): The dimension of the input features.
            hidden_dim (int): The dimension of each hidden layer.
            output_dim (int): The dimension of the output features.
            num_layers (int): The number of gated hidden layers. Must be at least 1.
        """
        super(DeepGatedMLP, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1 for DeepGatedMLP.")

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()

        # First hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.gate_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.gate_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the DeepGatedMLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: A tuple containing:
                - out (torch.Tensor): The final output of the DeepGatedMLP.
                - all_gate_vals (list[torch.Tensor]): A list of computed gate values
                                                      for each hidden layer.
        """
        all_gate_vals = []
        h = x
        for i in range(self.num_layers):
            # Pass through linear layer and apply ReLU activation
            h_pre_activation = self.layers[i](h)
            h_activated = F.relu(h_pre_activation)

            # Compute gate values for the current layer
            gate_val = torch.sigmoid(self.gate_layers[i](h_activated))
            all_gate_vals.append(gate_val)

            # Apply gating: element-wise multiplication
            h = h_activated * gate_val

        # Pass the final gated hidden representation through the output layer
        out = self.output_layer(h)
        return out, all_gate_vals

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
    and then uses a DeepGatedMLP for final classification.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, num_transformer_blocks: int = 12,
                 num_heads: int = 12, mlp_dim_ratio: int = 4, num_classes: int = 100,
                 dropout_rate: float = 0.1, gated_mlp_num_layers: int = 2):
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
            gated_mlp_num_layers (int): Number of hidden layers in the DeepGatedMLP head.
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

        # 4. Classification Head: DeepGatedMLP and a standard MLP in parallel
        # The diagram shows both GateMLP and MLP leading to the output after Majority Voting.
        # We'll use the DeepGatedMLP as the primary classification head.
        self.gated_mlp_head = DeepGatedMLP(
            input_dim=embed_dim,
            hidden_dim=self.mlp_dim, # Can be adjusted
            output_dim=num_classes,
            num_layers=gated_mlp_num_layers # Use the new parameter
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

        # 5. Classification with DeepGatedMLP head
        # The diagram implies these are final classification layers.
        # We return the output from the DeepGatedMLP head.
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
