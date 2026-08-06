import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import List, Sequence, Tuple
from .upsample import PatchExpand, BasicLayer_up,finalExpand_featuremap


class SwinDecoder(nn.Module):
    """
    Swin-Unet decoder.

    Input:
        bottleneck:
            Final encoder output.

        skip_features:
            Multi-scale encoder features.

    Output:
        High-resolution feature map:
            [B, embed_dim, image_height, image_width]
    """

    def __init__(
        self,
        patches_resolution: Tuple[int, int],
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
        use_checkpoint: bool = False,
        final_featuremap: str = "upsample_to_96",
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
        self.patches_resolution = tuple(patches_resolution)
        self.final_featuremap = final_featuremap

        drop_path_values = [
            value.item()
            for value in torch.linspace(
                0,
                drop_path_rate,
                sum(depths),
            )
        ]

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for decoder_index in range(self.num_layers):
            encoder_stage_index = (
                self.num_layers - 1 - decoder_index
            )

            stage_dim = int(
                embed_dim * 2 ** encoder_stage_index
            )

            stage_resolution = (
                self.patches_resolution[0]
                // (2 ** encoder_stage_index),
                self.patches_resolution[1]
                // (2 ** encoder_stage_index),
            )

            # The first decoder layer receives only the bottleneck.
            # Later layers concatenate encoder skip features.
            if decoder_index == 0:
                concat_projection = nn.Identity()

                layer_up = PatchExpand(
                    input_resolution=stage_resolution,
                    dim=stage_dim,
                    dim_scale=2,
                    norm_layer=norm_layer,
                )

            else:
                concat_projection = nn.Linear(
                    2 * stage_dim,
                    stage_dim,
                )

                depth_start = sum(
                    depths[:encoder_stage_index]
                )
                depth_end = sum(
                    depths[:encoder_stage_index + 1]
                )

                layer_up = BasicLayer_up(
                    dim=stage_dim,
                    input_resolution=stage_resolution,
                    depth=depths[encoder_stage_index],
                    num_heads=num_heads[
                        encoder_stage_index
                    ],
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
                    upsample=(
                        PatchExpand
                        if decoder_index
                        < self.num_layers - 1
                        else None
                    ),
                    use_checkpoint=use_checkpoint,
                )

            self.concat_back_dim.append(
                concat_projection
            )
            self.layers_up.append(layer_up)

        self.norm_up = norm_layer(embed_dim)

        if final_featuremap != "upsample_to_96":
            raise ValueError(
                "Only final_featuremap='upsample_to_96' "
                f"is supported, got {final_featuremap!r}."
            )

        # This custom class must return BCHW:
        # [B, embed_dim, image_height, image_width]
        self.up = finalExpand_featuremap(
            input_resolution=self.patches_resolution,
            dim_scale=4,
            dim=embed_dim,
        )

    def forward_tokens(
        self,
        x: torch.Tensor,
        skip_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode bottleneck tokens to the highest patch resolution.

        Returns:
            Token tensor [B, patch_H * patch_W, embed_dim].
        """
        if len(skip_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} skip features, "
                f"got {len(skip_features)}."
            )

        for decoder_index, layer_up in enumerate(
            self.layers_up
        ):
            if decoder_index == 0:
                x = layer_up(x)
                continue

            # Generalized replacement for x_downsample[3 - inx].
            skip_index = (
                self.num_layers - 1 - decoder_index
            )
            skip = skip_features[skip_index]

            if x.shape != skip.shape:
                raise RuntimeError(
                    "Decoder and skip feature shapes do not match: "
                    f"decoder={tuple(x.shape)}, "
                    f"skip={tuple(skip.shape)}, "
                    f"decoder_index={decoder_index}, "
                    f"skip_index={skip_index}."
                )

            x = torch.cat(
                [x, skip],
                dim=-1,
            )

            x = self.concat_back_dim[
                decoder_index
            ](x)

            x = layer_up(x)

        x = self.norm_up(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        skip_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
            High-resolution feature map:
            [B, embed_dim, image_height, image_width]
        """
        x = self.forward_tokens(
            x,
            skip_features,
        )

        x = self.up(x)

        return x
    