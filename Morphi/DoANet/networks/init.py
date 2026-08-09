from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .encoder import SwinEncoder
from .decoder import SwinDecoder


def init_swin_weights(module: nn.Module) -> None:
    """Initialize Linear and LayerNorm modules in Swin style."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
        
        
class Encoder_decoder_swin(nn.Module):
    """
    Encoder-decoder Swin model.

    By default, forward() returns a high-resolution feature map:
        [B, embed_dim, H, W]

    If output_logits=True, it returns:
        [B, num_classes, H, W]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
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
        final_featuremap: str = "upsample_to_96",
        output_logits: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.output_logits = output_logits

        self.encoder = SwinEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )

        self.decoder = SwinDecoder(
            patches_resolution=(
                self.encoder.patches_resolution
            ),
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            final_featuremap=final_featuremap,
        )

        # Optional segmentation/classification map head.
        if output_logits:
            self.output = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=num_classes,
                kernel_size=1,
                bias=False,
            )
        else:
            self.output = nn.Identity()

        # Initialize the entire model only once.
        self.apply(init_swin_weights)

        if (
            self.encoder.absolute_pos_embed
            is not None
        ):
            trunc_normal_(
                self.encoder.absolute_pos_embed,
                std=0.02,
            )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return high-resolution decoder feature map.

        Shape:
            [B, embed_dim, H, W]
        """
        bottleneck, skip_features = self.encoder(x)

        feature_map = self.decoder(
            bottleneck,
            skip_features,
        )

        return feature_map

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        feature_map = self.forward_features(x)

        output = self.output(feature_map)

        return output

    @torch.jit.ignore
    def no_weight_decay(self):
        # Parameter name changed after introducing encoder prefix.
        return {
            "encoder.absolute_pos_embed"
        }

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {
            "relative_position_bias_table"
        }