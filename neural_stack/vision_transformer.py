import torch
import torch.nn as nn

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
        image_patches = self.proj(images)           # [B, embed_dim, H/P, W/P]
        image_patches = image_patches.flatten(2).transpose(-2, -1)    # [B, num_patches, embed_dim]
        
        image_patches = torch.concat((self.cls_token, image_patches), dim=1)    # [B, num_patches + 1, embed_dim]
        image_patches = image_patches + self.pos_embedding                      # [B, num_patches + 1, embed_dim]

        return image_patches