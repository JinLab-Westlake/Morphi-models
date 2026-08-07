# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
import torch.nn as nn
from .init import Encoder_decoder_swin



logger = logging.getLogger(__name__)




class DoANet(nn.Module):
    """
    Outputs:
        seg_logits:
            [B, num_classes, H, W]

        death_prob:
            [B, 1, H, W], range [0, 1]
    """

    def __init__(
        self,
        config,
        img_size=224,
        num_classes=3,
        zero_head=False,
        vis=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = int(
            config.MODEL.SWIN.EMBED_DIM
        )

        self.backbone = Encoder_decoder_swin(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            attn_drop_rate=getattr(
                config.MODEL,
                "ATTN_DROP_RATE",
                0.0,
            ),
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            final_featuremap="upsample_to_96",
            output_logits=False,
        )

        # Multi-class segmentation head
        self.seg_head = nn.Conv2d(
            in_channels=self.feature_dim,
            out_channels=num_classes,
            kernel_size=1,
        )

        # Death-prob head
        self.death_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.feature_dim,
                out_channels=1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected BCHW input, got {tuple(x.shape)}"
            )

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x)

        seg_logits = self.seg_head(features)
        death_prob = self.death_head(features)

        return seg_logits, death_prob
    
    def load_from(self, config):
        """
        Load Swin Transformer pretrained weights into backbone.encoder only.

        Randomly initialized:
            backbone.decoder
            seg_head
            death_head
        """
        pretrained_path = config.MODEL.PRETRAIN_CKPT

        if not pretrained_path:
            print("No pretrained checkpoint. Train DoaNet from scratch.")
            return

        print(f"Loading pretrained Swin encoder from: {pretrained_path}")

        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
        )

        # Compatible with common checkpoint formats
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            pretrained_state = checkpoint["model"]

        elif (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
        ):
            pretrained_state = checkpoint["state_dict"]

        else:
            pretrained_state = checkpoint

        if not isinstance(pretrained_state, dict):
            raise TypeError(
                "The pretrained checkpoint does not contain a valid state_dict."
            )

        encoder_state = self.backbone.encoder.state_dict()

        loadable_state = {}
        shape_mismatch = []
        ignored_keys = []

        def clean_key(key):
            """
            Remove possible wrapper prefixes.

            Examples:
                module.patch_embed.proj.weight
                    -> patch_embed.proj.weight

                backbone.encoder.layers.0...
                    -> layers.0...

                swin_unet.layers.0...
                    -> layers.0...
            """
            prefixes = (
                "module.",
                "model.",
                "network.",
                "net.",
                "swin_unet.",
                "backbone.encoder.",
                "backbone.",
                "encoder.",
            )

            changed = True

            while changed:
                changed = False

                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
                        break

            return key

        for source_key, value in pretrained_state.items():
            if not torch.is_tensor(value):
                continue

            target_key = clean_key(source_key)

            # only load encoder actual param
            if target_key not in encoder_state:
                ignored_keys.append(source_key)
                continue

            # Params size must consistent
            if value.shape != encoder_state[target_key].shape:
                shape_mismatch.append(
                    (
                        source_key,
                        target_key,
                        tuple(value.shape),
                        tuple(encoder_state[target_key].shape),
                    )
                )
                continue

            loadable_state[target_key] = value

        result = self.backbone.encoder.load_state_dict(
            loadable_state,
            strict=False,
        )

        loaded_numel = sum(
            value.numel()
            for value in loadable_state.values()
        )

        total_numel = sum(
            value.numel()
            for value in encoder_state.values()
        )

        loaded_ratio = loaded_numel / max(total_numel, 1)

        print(
            f"Loaded encoder parameters: "
            f"{len(loadable_state)} tensors, "
            f"{loaded_ratio:.2%} parameter elements."
        )

        if shape_mismatch:
            print(
                f"Skipped {len(shape_mismatch)} parameters "
                "because of shape mismatch:"
            )

            for (
                source_key,
                target_key,
                source_shape,
                target_shape,
            ) in shape_mismatch[:20]:
                print(
                    f"  {source_key}: "
                    f"{source_shape} -> {target_key}: {target_shape}"
                )

        if result.missing_keys:
            print(
                f"Encoder missing keys: "
                f"{len(result.missing_keys)}"
            )

            for key in result.missing_keys[:20]:
                print(f"  missing: {key}")

        if result.unexpected_keys:
            print(
                f"Encoder unexpected keys: "
                f"{len(result.unexpected_keys)}"
            )

            for key in result.unexpected_keys[:20]:
                print(f"  unexpected: {key}")

        if loaded_ratio < 0.70:
            raise RuntimeError(
                "Too few encoder parameters were loaded "
                f"({loaded_ratio:.2%}). Check whether the checkpoint "
                "matches the current embed_dim, depths, num_heads, "
                "patch_size and window_size."
            )