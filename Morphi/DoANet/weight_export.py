import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from typing import Optional
import types
import os
import yaml

from networks.vision_transformer import SwinUnet



def _to_ns(d):
    if isinstance(d, dict):
        ns = types.SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, _to_ns(v))
        return ns
    elif isinstance(d, list):
        return [_to_ns(x) for x in d]
    else:
        return d

# config setting
def load_config(yaml_path: str):
    cfg = yaml.safe_load(open(yaml_path, "r"))
    cfg.setdefault("DATA", {}) ; cfg["DATA"].setdefault("IMG_SIZE", 224)
    cfg.setdefault("TRAIN", {}) ; cfg["TRAIN"].setdefault("USE_CHECKPOINT", False)
    cfg.setdefault("MODEL", {})
    cfg["MODEL"].setdefault("DROP_RATE", 0.0)
    cfg["MODEL"].setdefault("DROP_PATH_RATE", 0.0)
    cfg["MODEL"].setdefault("PRETRAIN_CKPT", None)
    cfg["MODEL"].setdefault("SWIN", {})
    sw = cfg["MODEL"]["SWIN"]
    sw.setdefault("PATCH_SIZE", 4)
    sw.setdefault("IN_CHANS", 3)     #rgb
    sw.setdefault("EMBED_DIM", 96)
    sw.setdefault("DEPTHS", [2,2,2,2])
    sw.setdefault("NUM_HEADS", [3,6,12,24])
    sw.setdefault("WINDOW_SIZE", 7)
    sw.setdefault("MLP_RATIO", 4.0)
    sw.setdefault("QKV_BIAS", True)
    sw.setdefault("QK_SCALE", None)
    sw.setdefault("APE", False)
    sw.setdefault("PATCH_NORM", True)
    return _to_ns(cfg)

# Size adaptation wrapper
def _ceil_to_m(x, m)
    return ((x + m - 1)//m)*m


class FixedPadToImgSize(nn.Module):
    def __init__(self, backbone: nn.Module, cfg):
        super().__init__()
        self.backbone = backbone.eval()
        self.Ht = int(cfg.DATA.IMG_SIZE)
        self.Wt = int(cfg.DATA.IMG_SIZE)

    @torch.no_grad()
    def forward(self, x):             # x: (N,C,H,W)
        N, C, H, W = x.shape
        pad_h = max(0, self.Ht - H)
        pad_w = max(0, self.Wt - W)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)   # (left,right,top,bottom)

        out = self.backbone(x)        # model return (logits, prob)

        # crop into orignal size
        if isinstance(out, (tuple, list)):
            return tuple(o[..., :H, :W] for o in out)
        else:
            return out[..., :H, :W]




# weight load
def load_weights(model: nn.Module, cfg, seg_ckpt: Optional[str]):
    if seg_ckpt and os.path.isfile(seg_ckpt):
        state = torch.load(seg_ckpt, map_location="cpu")
        if isinstance(state, dict):
            for k in ["state_dict", "model", "net", "module", "ema", "generator"]:
                if k in state and isinstance(state[k], dict):
                    state = state[k]; break
        state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:   print("[load] missing keys:", missing[:10], "...")
        if unexpected:print("[load] unexpected keys:", unexpected[:10], "...")
    else:
        model.load_from(cfg)  # Only load encoder pre-train

def export_onnx(cfg_path, seg_ckpt, out_path="model.onnx",
                num_classes=3, dummy_hw=(512,512), opset=17):
    cfg = load_config(cfg_path)

    # build model
    model = SwinUnet(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=num_classes, zero_head=False, vis=False)
    model.eval()
    load_weights(model, cfg, seg_ckpt)

    wrapped = FixedPadToImgSize(model, cfg).eval()

    C = cfg.MODEL.SWIN.IN_CHANS
    H, W = dummy_hw
    dummy = torch.randn(1, C, H, W, dtype=torch.float32)

    torch.onnx.export(
        wrapped, dummy, out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits", "prob"],  # two output
        dynamic_axes={
            "image":      {0:"N", 2:"H", 3:"W"},
            "logits":     {0:"N", 2:"H", 3:"W"},
            "prob": {0:"N", 2:"H", 3:"W"},
        },
    )
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    print(f"[export] OK -> {out_path} (opset={opset})")



if __name__ == "__main__":
    # set the path
    CFG  = "cfg.yaml"
    CKPT = "epoch.pth"
    export_onnx(CFG, CKPT, out_path="model.onnx", dummy_hw=(224,224))
