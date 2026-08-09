import sys, torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import TracingAdapter

sys.path.insert(0, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/swinb")
from config import add_swinb_config #import add config weight


class MaskLogitsStub(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.m = model
    def forward(self, x):
        # input
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x[:1]
        N, C, H, W = x.shape

        # detectron2 forward（to features/proposals/box）
        bi = [{"image": x[0], "height": H, "width": W}]
        images = self.m.preprocess_image(bi)            # ImageList
        feats  = self.m.backbone(images.tensor)

        # proposals：Compatible with 1/2 returns
        p_out = self.m.proposal_generator(images, feats, None)
        proposals = p_out[0] if isinstance(p_out, (tuple, list)) else p_out

        # box: compatible with 1/2 returns; Some versions return Instances directly
        b_out = self.m.roi_heads._forward_box(feats, proposals)
        pred_insts = b_out[0] if isinstance(b_out, (tuple, list)) else b_out
        inst = pred_insts if hasattr(pred_insts, "get_fields") else pred_insts[0]

        # If there are no results, use proposals
        if len(inst) == 0:
            p = proposals[0].proposal_boxes.tensor[:1]
            boxes   = p
            scores  = p.new_zeros((1,), dtype=torch.float32)
            classes = torch.zeros((1,), dtype=torch.long, device=p.device)

            if getattr(self.m.roi_heads, "mask_on", False):
                from detectron2.structures import Boxes
                mf = self.m.roi_heads.mask_pooler(
                    [feats[f] for f in self.m.roi_heads.in_features],
                    [Boxes(p)]
                )
                # Only convolutional prediction, no instances, no pasting/thresholding
                mask_logits = self.m.roi_heads.mask_head.layers(mf)     # [1,C,M,M]
            else:
                M = 28
                mask_logits = p.new_zeros((1, 1, M, M), dtype=torch.float32)

            # Binding minimal dependencies to ensure that the exported graph "uses" the input
            eps = x.mean().to(torch.float32) * 1e-8
            boxes = boxes + eps
            scores = scores + eps
            mask_logits = mask_logits + eps
            # classes is int64
            return boxes.contiguous(), scores.contiguous(), classes.contiguous(), mask_logits.contiguous()

        # There are normal test results
        boxes   = inst.pred_boxes.tensor
        scores  = inst.scores
        classes = inst.pred_classes.to(torch.int64)

        if getattr(self.m.roi_heads, "mask_on", False):
            mf = self.m.roi_heads.mask_pooler(
                [feats[f] for f in self.m.roi_heads.in_features],
                [inst.pred_boxes]
            )
            mask_logits = self.m.roi_heads.mask_head.layers(mf)         # [N,C,M,M]
        else:
            M = 28
            mask_logits = boxes.new_zeros((0, 1, M, M), dtype=torch.float32)

        # Also do minimal dependency binding + continuity to improve export stability
        eps = x.mean().to(torch.float32) * 1e-8
        boxes = boxes + eps
        scores = scores + eps
        mask_logits = mask_logits + eps
        return boxes.contiguous(), scores.contiguous(), classes.contiguous(), mask_logits.contiguous()







class ONNXFriendly(torch.nn.Module):
    """
    接收 x: (N,3,H,W)；内部构造 batched_inputs；输出纯 Tensor（boxes/scores/classes/masks）
    先固定 N=1，导出成功后再考虑扩展动态 batch。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (N,3,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        N, C, H, W = x.shape
        assert N == 1, "先固定 batch=1；需要的话再做动态 batch 导出"
        batched_inputs = [{"image": x[0], "height": H, "width": W}]
        outputs = self.model(batched_inputs)   # list[dict], 推理模式返回 Instances

        # 展平 Instances -> 纯 tensor
        out = outputs[0]
        inst = out["instances"]
        boxes   = inst.pred_boxes.tensor      # (M,4) xyxy float32
        scores  = inst.scores                 # (M,)
        classes = inst.pred_classes.to(torch.int64)  # (M,)
        # mask：如果没有 mask 分支，给一个空 tensor
        masks = getattr(inst, "pred_masks", None)
        if masks is None:
            masks = boxes.new_zeros((0, H, W))  # (0,H,W)
        else:
            masks = masks.float()               # (M,H,W) 概率/二值按你模型而定

        return boxes, scores, classes, masks
    
    
    

@torch.no_grad()
def to_onnx():
    cfg = get_cfg()
    add_swinb_config(cfg)  # regist self keys

    # 容忍历史键（like FPN.TOP_LEVELS），avoid KeyError
    cfg.set_new_allowed(True)
    cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/config.yaml")
    cfg.set_new_allowed(False)
    
    # make sure include mask head in export
    cfg.MODEL.MASK_ON = True
    
    # make sure dummy also have box
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.freeze()


    # # 检查 backbone 是否已注册
    # from detectron2.modeling.backbone import BACKBONE_REGISTRY
    # assert cfg.MODEL.BACKBONE.NAME in BACKBONE_REGISTRY._obj_map, f"Backbone not registered: {cfg.MODEL.BACKBONE.NAME}"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # wrapper = ONNXFriendly(model).eval()
    
    stub = MaskLogitsStub(model).eval()

    H = W = 800  # size in inference
    dummy = torch.randn(1, 3, H, W)


    # outs = wrapper(dummy)
    # if not isinstance(outs, (tuple, list)):
    #     outs = (outs,)
    # output_names = [f"out{i}" for i in range(len(outs))]   # 依次为 boxes/scores/classes/mask_logits

    # torch.onnx.export(
    #     wrapper, dummy, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/Export/model.onnx",
    #     opset_version=16,             
    #     do_constant_folding=False,
    #     input_names=["input"],
    #     output_names=output_names,
    #     dynamic_axes={
    #         # 我们固定 batch=1，这里只给 H/W 动态；需要的话你再扩 batch
    #         "input": {2: "h", 3: "w"},
    #         output_names[0]: {0: "num_boxes"},  # boxes
    #         output_names[1]: {0: "num_boxes"},  # scores
    #         output_names[2]: {0: "num_boxes"},  # classes
    #         output_names[3]: {0: "num_boxes"},  # mask_logits 的 N_inst 维
    #     }
    # )
    # print("Exported to model.onnx")
    
    # model = build_model(cfg); DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS); model.eval()
    # stub = MaskLogitsStub(model).eval()
    # dummy = torch.randn(1,3,800,800)

    torch.onnx.export(
        stub, dummy, "model_masklogits_op13.onnx",
        opset_version=13, do_constant_folding=False,
        input_names=["input"], output_names=["boxes","scores","classes","mask_logits"],
        dynamic_axes={
            "input": {2:"h", 3:"w"},
            "boxes": {0:"num_boxes"},
            "scores": {0:"num_boxes"},
            "classes": {0:"num_boxes"},
            "mask_logits": {0:"num_boxes"},
        }
    )

    print("✅ exported model_masklogits_op13.onnx")

if __name__ == "__main__":
    to_onnx()