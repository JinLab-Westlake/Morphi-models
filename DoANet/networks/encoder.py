import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import List, Sequence, Tuple
from .swin_transformer_block import PatchEmbed,BasicLayer,PatchMerging



class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder.

    Returns:
        bottleneck:
            Final encoder feature, shape [B, L_bottom, C_bottom].

        skip_features:
            Features saved before every encoder stage.
            For the default four-stage model:

            skip_features[0]: [B, 56*56, 96]
            skip_features[1]: [B, 28*28, 192]
            skip_features[2]: [B, 14*14, 384]
            skip_features[3]: [B, 7*7, 768]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        if len(depths) != len(num_heads):
            raise ValueError(
                "depths and num_heads must contain the same number "
                f"of stages, got {len(depths)} and {len(num_heads)}."
            )

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)

        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self.num_features = int(
            embed_dim * 2 ** (self.num_layers - 1)
        )

        # Image -> patch tokens
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.patches_resolution = tuple(
            self.patch_embed.patches_resolution
        )
        self.num_patches = self.patch_embed.num_patches

        # Optional absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.num_patches,
                    embed_dim,
                )
            )
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Encoder DropPath schedule
        drop_path_values = [
            value.item()
            for value in torch.linspace(
                0,
                drop_path_rate,
                sum(depths),
            )
        ]

        self.layers = nn.ModuleList()

        for stage_index in range(self.num_layers):
            stage_dim = int(
                embed_dim * 2 ** stage_index
            )

            stage_resolution = (
                self.patches_resolution[0]
                // (2 ** stage_index),
                self.patches_resolution[1]
                // (2 ** stage_index),
            )

            depth_start = sum(depths[:stage_index])
            depth_end = sum(depths[:stage_index + 1])

            layer = BasicLayer(
                dim=stage_dim,
                input_resolution=stage_resolution,
                depth=depths[stage_index],
                num_heads=num_heads[stage_index],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_values[
                    depth_start:depth_end
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging
                    if stage_index < self.num_layers - 1
                    else None
                ),
                use_checkpoint=use_checkpoint,
            )

            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Image tensor [B, in_chans, H, W].

        Returns:
            bottleneck:
                [B, L_bottom, C_bottom]

            skip_features:
                Multi-scale token features.
        """
        x = self.patch_embed(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        skip_features: List[torch.Tensor] = []

        for layer in self.layers:
            # Preserve the feature before the current stage.
            skip_features.append(x)
            x = layer(x)

        x = self.norm(x)

        return x, skip_features