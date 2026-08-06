import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from .swin_transformer_block import SwinTransformerBlock



import torch
import torch.nn as nn


class PatchExpand(nn.Module):
    """
    Token-space 2D pixel-shuffle upsampling.

    Input:
        x: [B, H*W, C]

    Output when scale_factor=2:
        x: [B, (2H)*(2W), C/2]
    """

    def __init__(
        self,
        input_resolution,
        dim,
        dim_scale=2,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        if len(input_resolution) != 2:
            raise ValueError(
                "input_resolution must be (H, W), "
                f"got {input_resolution}."
            )

        if not isinstance(dim_scale, int) or dim_scale < 1:
            raise ValueError(
                f"dim_scale must be a positive integer, got {dim_scale}."
            )

        self.input_resolution = tuple(input_resolution)
        self.dim = int(dim)
        self.dim_scale = int(dim_scale)

        # Swin-Unet 的标准 PatchExpand:
        # scale=2 时，输出通道从 C 降为 C/2。
        if self.dim % self.dim_scale != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by "
                f"dim_scale={self.dim_scale}."
            )

        self.output_dim = self.dim // self.dim_scale

        # PixelShuffle 要求输入通道数为:
        # output_dim * scale_factor^2
        expanded_dim = (
            self.output_dim
            * self.dim_scale
            * self.dim_scale
        )

        # scale=2 时:
        # expanded_dim = C/2 * 4 = 2C
        #
        # 与原始实现 nn.Linear(C, 2*C) 完全一致。
        self.expand = nn.Linear(
            self.dim,
            expanded_dim,
            bias=False,
        )

        self.pixel_shuffle = nn.PixelShuffle(
            upscale_factor=self.dim_scale
        )

        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        Args:
            x: [B, H*W, C]

        Returns:
            x: [
                B,
                (H*scale)*(W*scale),
                C/scale
            ]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected [B, L, C], got {tuple(x.shape)}."
            )

        H, W = self.input_resolution
        B, L, C = x.shape

        if L != H * W:
            raise ValueError(
                f"Token length mismatch: L={L}, "
                f"but H*W={H * W}."
            )

        if C != self.dim:
            raise ValueError(
                f"Channel mismatch: expected {self.dim}, got {C}."
            )

        # [B, H*W, C]
        # -> [B, H*W, output_dim * scale^2]
        x = self.expand(x)

        # Token -> feature map
        # [B, H*W, C_expanded]
        # -> [B, C_expanded, H, W]
        x = x.transpose(1, 2).contiguous()
        x = x.view(
            B,
            self.output_dim * self.dim_scale**2,
            H,
            W,
        )

        # PixelShuffle:
        # [B, output_dim*scale^2, H, W]
        # -> [B, output_dim, H*scale, W*scale]
        x = self.pixel_shuffle(x)

        H_out = H * self.dim_scale
        W_out = W * self.dim_scale

        # Feature map -> token
        # [B, output_dim, H_out, W_out]
        # -> [B, H_out*W_out, output_dim]
        x = x.flatten(2).transpose(1, 2).contiguous()

        expected_shape = (
            B,
            H_out * W_out,
            self.output_dim,
        )

        if tuple(x.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected PatchExpand output shape: "
                f"expected {expected_shape}, "
                f"got {tuple(x.shape)}."
            )

        x = self.norm(x)

        return x

    def extra_repr(self):
        return (
            f"input_resolution={self.input_resolution}, "
            f"dim={self.dim}, "
            f"dim_scale={self.dim_scale}, "
            f"output_dim={self.output_dim}"
        )
        
        

class finalExpand_featuremap(nn.Module):
    def __init__(
        self,
        input_resolution,
        dim,
        dim_scale=4,
        out_dim=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.out_dim = out_dim or dim

        # 每个 token 展开成 dim_scale × dim_scale 个空间位置
        self.expand = nn.Linear(
            dim,
            (dim_scale ** 2) * self.out_dim,
            bias=False,
        )

        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        """
        Args:
            x: [B, H*W, dim]

        Returns:
            x: [B, out_dim, H*scale, W*scale]
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, (
            f"Token number {L} does not match "
            f"input resolution {H}x{W}"
        )
        assert C == self.dim, (
            f"Input channel {C} does not match dim={self.dim}"
        )

        # [B, H*W, dim]
        # -> [B, H*W, scale²*out_dim]
        x = self.expand(x)

        # -> [B, H, W, scale²*out_dim]
        x = x.view(
            B,
            H,
            W,
            (self.dim_scale ** 2) * self.out_dim,
        )

        # PixelShuffle 式空间重排
        # -> [B, H*scale, W*scale, out_dim]
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=self.out_dim,
        )

        x = self.norm(x)

        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        return x
    
#add
class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

