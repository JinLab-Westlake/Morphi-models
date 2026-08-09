import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import Conv2d, ConvTranspose2d
from detectron2.modeling.roi_heads.mask_head import (
    ROI_MASK_HEAD_REGISTRY,
    BaseMaskRCNNHead,
    mask_rcnn_loss,
    mask_rcnn_inference,
)
# from detectron2.utils.registry import Registry
from fvcore.nn import weight_init


@ROI_MASK_HEAD_REGISTRY.register()
class EdgeRefineMaskHead(BaseMaskRCNNHead):
    """
    Mask head with coarse mask prediction and learnable refinement.

    Structure:
        ROI feature, usually 14×14
            ↓
        Conv ×4
            ↓
        Deconv, upsample to 28×28
            ↓
        Coarse mask logits
            ↓
        concat(28×28 mask feature, coarse mask logits)
            ↓
        Refinement convs
            ↓
        Refined mask logits

    During training:
        loss = refined mask loss + lambda_coarse * coarse mask loss

    During inference:
        only refined mask logits are used.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.num_mask_classes = 1 if self.cls_agnostic_mask else self.num_classes

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV

        self.coarse_loss_weight = 0.5

        input_channels = input_shape.channels

        # Shared conv layers: same idea as standard Mask R-CNN mask head
        self.mask_fcn = []
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=F.relu,
            )
            self.add_module(f"mask_fcn{k + 1}", conv)
            self.mask_fcn.append(conv)

        # 14×14 → 28×28
        self.deconv = ConvTranspose2d(
            conv_dim,
            conv_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            # activation=F.relu,
        )

        # Coarse mask predictor
        self.coarse_predictor = Conv2d(
            conv_dim,
            self.num_mask_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Refinement module
        # Input = 28×28 mask feature + 28×28 coarse mask logits
        refine_in_channels = conv_dim + self.num_mask_classes

        self.refine_conv1 = Conv2d(
            refine_in_channels,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=F.relu,
        )

        self.refine_conv2 = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=F.relu,
        )

        self.refined_predictor = Conv2d(
            conv_dim,
            self.num_mask_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Weight initialization
        for layer in self.mask_fcn:
            weight_init.c2_msra_fill(layer)

        weight_init.c2_msra_fill(self.deconv)
        weight_init.c2_msra_fill(self.refine_conv1)
        weight_init.c2_msra_fill(self.refine_conv2)

        nn.init.normal_(self.coarse_predictor.weight, std=0.001)
        nn.init.constant_(self.coarse_predictor.bias, 0)

        nn.init.normal_(self.refined_predictor.weight, std=0.001)
        nn.init.constant_(self.refined_predictor.bias, 0)

    def layers(self, x):
        """
        x: ROIAlign output, usually [N, 256, 14, 14]

        returns:
            coarse_logits:  [N, C, 28, 28]
            refined_logits: [N, C, 28, 28]
        """

        for layer in self.mask_fcn:
            x = layer(x)

        # mask_feature = self.deconv(x)  # [N, 256, 28, 28]
        mask_feature = F.relu(self.deconv(x))  # [N, 256, 28, 28]

        coarse_logits = self.coarse_predictor(mask_feature)

        refine_input = torch.cat([mask_feature, coarse_logits], dim=1)

        refine_feature = self.refine_conv1(refine_input)
        refine_feature = self.refine_conv2(refine_feature)

        refined_logits = self.refined_predictor(refine_feature)

        return coarse_logits, refined_logits

    def forward(self, x, instances):
        """
        During training:
            return loss dict

        During inference:
            attach pred_masks to instances
        """

        coarse_logits, refined_logits = self.layers(x)

        if self.training:
            loss_refined = mask_rcnn_loss(
                refined_logits,
                instances,
                self.vis_period,
            )

            loss_coarse = mask_rcnn_loss(
                coarse_logits,
                instances,
                self.vis_period,
            )

            return {
                "loss_mask": loss_refined,
                "loss_mask_coarse": self.coarse_loss_weight * loss_coarse,
            }

        else:
            mask_rcnn_inference(refined_logits, instances)
            return instances