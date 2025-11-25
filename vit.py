"""Simple Vision Transformer (ViT) variants for CIFAR-10.

This module provides a lightweight ViT implementation and a few factory
functions (names starting with "vit") so the existing `train.py` helper will
discover them automatically.

The implementations are tuned for 32x32 CIFAR input (small patch sizes).
These are not production-grade (no attention optimizations), but are
easy-to-read and intended to have comparable total parameter counts /
computation to the plain/resnet variants in this repo.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchEmbed(nn.Module):
    """Image to patch embeddings using a conv layer.

    Input: (B, 3, H, W)
    Output: (B, N_patches, embed_dim)
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: B x C x H x W
        x = self.proj(x)  # B x embed_dim x Gh x Gw
        x = x.flatten(2)  # B x embed_dim x N
        x = x.transpose(1, 2)  # B x N x embed_dim
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        # x: B x N x E
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_chans=3,
                 num_classes=10,
                 embed_dim=192,
                 depth=6,
                 num_heads=3,
                 mlp_ratio=4.0,
                 dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # transformer encoder
        layers = []
        for _ in range(depth):
            layers.append(TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout))
        self.encoder = nn.Sequential(*layers)

        # classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # head init
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B x N x E

        cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x E
        x = torch.cat((cls_tokens, x), dim=1)  # B x (N+1) x E
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.encoder(x)
        x = self.norm(x)

        cls = x[:, 0]
        out = self.head(cls)
        return out


# Factory functions named with prefix 'vit' so train.py can detect them
def vit32(num_classes=10, dropout=0.0):
    """Variant roughly comparable to medium plainnets (balanced size).
    - patch 4 -> 8x8 patches -> 64 tokens
    - embed_dim 192, depth 6 -> modest parameter count
    """
    return VisionTransformer(patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0, dropout=dropout, num_classes=num_classes)


def vit56(num_classes=10, dropout=0.0):
    """Larger variant with more transformer layers and wider embedding.
    Intended to be closer to larger plainnet/resnet variants.
    """
    return VisionTransformer(patch_size=4, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4.0, dropout=dropout, num_classes=num_classes)


def vit110(num_classes=10, dropout=0.0):
    """Even larger variant for parity with the biggest plainnet/resnet.
    """
    return VisionTransformer(patch_size=4, embed_dim=384, depth=16, num_heads=8, mlp_ratio=4.0, dropout=dropout, num_classes=num_classes)


__all__ = ['VisionTransformer', 'vit32', 'vit56', 'vit110']


def test(net):
    """Print total params and layers similar to plainnet.test.

    This mirrors the helper in `plainnet.py` so it's easy to compare models.
    """
    total_params = 0
    for p in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(p.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == '__main__':
    print("Testing ViT variants for CIFAR-10:")
    print("=" * 60)

    for net_name in __all__:
        if net_name.startswith('vit'):
            print(f"\n{net_name}:")
            model = globals()[net_name]()
            test(model)

            # Test forward pass
            x = torch.randn(2, 3, 32, 32)
            y = model(x)
            print(f"Output shape: {y.shape}")

    print("\n" + "=" * 60)
    print("These ViT variants are intended to be comparable to the plainnet/resnet")
    print("implementations in this repo but adapted for CIFAR-10 (32x32).")
    print("=" * 60)
