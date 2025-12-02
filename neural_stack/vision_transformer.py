import torch
import torch.nn as nn

from neural_stack.attention import MultiHeadAttention
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        assert (img_size[0] * img_size[1]) % (patch_size ** 2) == 0, "Image dimensions must be divisible by the patch size."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] * img_size[1]) // patch_size ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply PatchEmbedding Layer

        Args:
            images (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: _description_
        """
        batch_size = images.shape[0]

        image_patches = self.proj(images)           # [B, embed_dim, H/P, W/P]
        image_patches = image_patches.flatten(2).transpose(-2, -1)    # [B, num_patches, embed_dim]
        
        image_patches = torch.concat((self.cls_token.repeat(batch_size, 1, 1), image_patches), dim=1)    # [B, num_patches + 1, embed_dim]
        image_patches = image_patches + self.pos_embedding                      # [B, num_patches + 1, embed_dim]

        return image_patches

class MLPBlock(nn.Module):
    def __init__(self, base_dim, expansion_ratio, activation='gelu', dropout=0.0):
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
    
    def forward(self, x):
        return self.block(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(TransformerBlock, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.multihead_selfattention = MultiHeadAttention(
            num_heads=num_heads,
            dim_model=embed_dim
        )

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.mlp_block = MLPBlock(embed_dim, mlp_ratio)

    def forward(self, x):
        x_norm = self.layer_norm_1(x)

        x_att_out, x_att_scores = self.multihead_selfattention(x_norm, x_norm, x_norm)
        x = x + x_att_out

        x = x + self.mlp_block(self.layer_norm_2(x))

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_ratio, dropout, num_classes):
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

    def forward(self, images):
        x = self.patch_embedding(images)

        for transformer_block in self.transformer_stack:
            x = transformer_block(x)
        
        cls_token = x[:, 0, :]
        out = self.classification_head(cls_token)

        return out