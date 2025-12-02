import torch
import torch.nn as nn
from typing import Tuple

from neural_stack.attention import MultiHeadAttention
class PatchEmbedding(nn.Module):
    """Patch Embedding layer for Vision Transformer.

    Splits an image into patches, projects them to embedding dimension,
    and adds positional embeddings and a class token.
    """

    def __init__(self, img_size: Tuple[int, int], patch_size: int, in_channels: int, embed_dim: int) -> None:
        """Initialize the patch embedding layer.

        Args:
            img_size: Image dimensions as (height, width).
            patch_size: Size of each square patch.
            in_channels: Number of input channels (e.g., 3 for RGB).
            embed_dim: Dimension of the embedding space.
        """
        super(PatchEmbedding, self).__init__()
        assert (img_size[0] * img_size[1]) % (patch_size ** 2) == 0, "Image dimensions must be divisible by the patch size."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] * img_size[1]) // patch_size ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply patch embedding to input images.

        Args:
            images: Input tensor of shape [B, C, H, W]

        Returns:
            Embedded patches with positional encoding and class token of shape [B, num_patches + 1, embed_dim]
        """
        batch_size = images.shape[0]

        image_patches = self.proj(images)           # [B, embed_dim, H/P, W/P]
        image_patches = image_patches.flatten(2).transpose(-2, -1)    # [B, num_patches, embed_dim]
        
        image_patches = torch.concat((self.cls_token.repeat(batch_size, 1, 1), image_patches), dim=1)    # [B, num_patches + 1, embed_dim]
        image_patches = image_patches + self.pos_embedding                      # [B, num_patches + 1, embed_dim]

        return image_patches

class MLPBlock(nn.Module):
    """Feed-forward MLP block with expansion and contraction.

    Applies two linear transformations with an activation function and dropout.
    Commonly used in transformer architectures.
    """

    def __init__(self, base_dim: int, expansion_ratio: float, activation: str = 'gelu', dropout: float = 0.0) -> None:
        """Initialize the MLP block.

        Args:
            base_dim: Input and output dimension.
            expansion_ratio: Ratio to expand hidden dimension (e.g., 4.0 for 4x expansion).
            activation: Activation function, either 'gelu' or 'relu'. Default: 'gelu'.
            dropout: Dropout probability. Default: 0.0.
        """
        super(MLPBlock, self).__init__()

        self.hidden_dim = int(base_dim * expansion_ratio)

        if activation == 'gelu':
            act = nn.GELU()
        else:
            act = nn.ReLU()

        self.block = nn.Sequential(
            nn.Linear(base_dim, self.hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, base_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP block transformation.

        Args:
            x: Input tensor of shape [B, N, base_dim]

        Returns:
            Transformed tensor of shape [B, N, base_dim]
        """
        return self.block(x)


class TransformerBlock(nn.Module):
    """Standard Transformer block with multi-head self-attention and MLP.

    Applies layer normalization, multi-head attention, and feed-forward MLP
    with residual connections.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float) -> None:
        """Initialize the transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for the MLP hidden dimension.
        """
        super(TransformerBlock, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.msa = MultiHeadAttention(
            num_heads=num_heads,
            dim_model=embed_dim
        )

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.mlp_block = MLPBlock(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block transformation.

        Args:
            x: Input tensor of shape [B, N, embed_dim]

        Returns:
            Transformed tensor of shape [B, N, embed_dim]
        """
        x_norm = self.layer_norm_1(x)

        x_att_out, x_att_scores = self.msa(x_norm, x_norm, x_norm)
        x = x + x_att_out

        x = x + self.mlp_block(self.layer_norm_2(x))

        return x
    
class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification.

    Implements the architecture from "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).
    Processes images as sequences of patches using transformer encoder blocks.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
        num_classes: int
    ) -> None:
        """Initialize the Vision Transformer.

        Args:
            img_size: Image dimensions as (height, width).
            patch_size: Size of each square patch.
            in_channels: Number of input channels (e.g., 3 for RGB).
            embed_dim: Dimension of the embedding space.
            num_heads: Number of attention heads in each transformer block.
            num_layers: Number of transformer blocks.
            mlp_ratio: Expansion ratio for MLP hidden dimension.
            dropout: Dropout probability.
            num_classes: Number of output classes for classification.
        """
        super(VisionTransformer, self).__init__()

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.transformer_stack = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )

        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Flatten(1),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply Vision Transformer to input images.

        Args:
            images: Input images of shape [B, C, H, W]

        Returns:
            Class logits of shape [B, num_classes]
        """
        x = self.patch_embedding(images)

        for transformer_block in self.transformer_stack:
            x = transformer_block(x)

        cls_token = x[:, 0, :]
        out = self.classification_head(cls_token)

        return out