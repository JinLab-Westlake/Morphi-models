#!/usr/bin/env python
# modified from Copyright (c) Facebook, Inc. and its affiliates.

"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import itertools 

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, DatasetMapper
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, BestCheckpointer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params

from swinb import add_swinb_config
from detectron2.data.datasets import register_coco_instances # register datasets
from detectron2.data.datasets import load_coco_json
from detectron2.engine import HookBase
import random
from refine_mask.edge_refine_mask_head import EdgeRefineMaskHead

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


# to valid the loss and others stuff
class ValidationLossHook(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        
        self.cfg.defrost() # Make it mutable again
        # Use the validation dataset for the internal loader
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST 
        self.cfg.freeze()  # Optional: freeze it again for safety
        
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        # Calculate loss every EVAL_PERIOD iterations
        if (self.trainer.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
            data = next(self._loader)
            with torch.no_grad():
                # Model returns a dict of losses when in training mode
                loss_dict = self.trainer.model(data)
                
                # Reduce across GPUs if using distributed training
                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss_dict_reduced.values())
                
                if comm.is_main_process():
                    # This puts 'total_val_loss' into TensorBoard/metrics.json
                    self.trainer.storage.put_scalars(total_val_loss=losses_reduced, **loss_dict_reduced)



class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # define train loader
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Overwriting the default loader to inject powerful enhancements.
        """
        augmentation_list = [
            T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"),
            T.RandomRotation(angle=[0, 90, 180, 270], sample_style="choice"),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),  # 仅在确实是彩色显微图时再开
        ]
        
        # Define the mapper with our list of powerful augmentations
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentation_list)
        
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={
                "absolute_pos_embed": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
                "relative_position_bias_table": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
            }
        )

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optimizer_type == "AdamW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer
    
    def build_hooks(self):
        hooks = super().build_hooks()
        
        # add validation loss hook
        hooks.append(ValidationLossHook(self.cfg))
        
        hooks.append(
                    BestCheckpointer(
                        eval_period=self.cfg.TEST.EVAL_PERIOD,
                        checkpointer=self.checkpointer,
                        val_metric="segm/AP", 
                        mode="max",
                        file_prefix="model_best"
                    )
                )
        return hooks





def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_swinb_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # cfg.SOLVER.CHECKPOINT_PERIOD = 50
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# def main(args):
#     cfg = setup(args)

#     if args.eval_only:
#         # create training model, and trainer is inherited from defaultTrainer
#         model = Trainer.build_model(cfg)
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 print("Trainable:",name)
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=args.resume
#         )
#         res = Trainer.test(cfg, model)
#         if cfg.TEST.AUG.ENABLED:
#             res.update(Trainer.test_with_TTA(cfg, model))
#         if comm.is_main_process():
#             verify_results(cfg, res)
#         return res

#     """
#     If you'd like to do anything fancier than the standard training logic,
#     consider writing your own training loop (see plain_train_net.py) or
#     subclassing the trainer.
#     """
#     trainer = Trainer(cfg)
#     trainer.resume_or_load(resume=args.resume)
#     if cfg.TEST.AUG.ENABLED:
#         trainer.register_hooks(
#             [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
#         )
#     return trainer.train()


# if __name__ == "__main__":
#     args = default_argument_parser().parse_args()
#     print("Command Line Args:", args)
    
#     # register_coco_instances("coco_my_train", {}, "/storage/jinxinLab/wangyimin/Cell_seg/detectron2/datasets/Droplets_datasets/train_coco_format.json", "/storage/jinxinLab/wangyimin/Cell_seg/detectron2/datasets/Droplets_datasets/train")
#     # register_coco_instances("coco_my_val", {}, "/storage/jinxinLab/wangyimin/Cell_seg/detectron2/datasets/Droplets_datasets/val_coco_format.json", "/storage/jinxinLab/wangyimin/Cell_seg/detectron2/datasets/Droplets_datasets/val")
    
#     # register_coco_instances("coco_my_train", {}, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train.json", "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train")
#     # register_coco_instances("coco_my_val", {}, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/val.json", "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/val")
    
#     # register_coco_instances("coco_my_train", {}, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/Organoids_finetune_Datasets/train.json", "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/Organoids_finetune_Datasets/train")
#     # register_coco_instances("coco_my_val", {}, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/Organoids_finetune_Datasets/val.json", "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/Organoids_finetune_Datasets/val")
    
#     # register_coco_instances("coco_my_train", {}, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train.json", "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train")
#     # register_coco_instances("coco_my_val", {}, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/val.json", "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/val")
    
    
    
#     # load data train:val = 8:2
#     json_file = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train.json"
#     image_root = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train"

#     # CHANGE: 'name' to 'dataset_name'
#     full_dataset_dicts = load_coco_json(json_file, image_root, dataset_name="full_data")

#     # Shuffle and split
#     random.seed(42)
#     random.shuffle(full_dataset_dicts)
#     split = int(0.8 * len(full_dataset_dicts))

#     train_dicts = full_dataset_dicts[:split]
#     val_dicts = full_dataset_dicts[split:]

#     # Register the split datasets
#     DatasetCatalog.register("coco_my_train", lambda: train_dicts)
#     MetadataCatalog.get("coco_my_train").set(
#         json_file=json_file, 
#         image_root=image_root, 
#         evaluator_type="coco"
#     )

#     DatasetCatalog.register("coco_my_val", lambda: val_dicts)
#     MetadataCatalog.get("coco_my_val").set(
#         json_file=json_file, 
#         image_root=image_root, 
#         evaluator_type="coco"
#     )
    
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )


def is_valid_coco_anno(anno):
    # 1. 必须有 segmentation 字段
    if "segmentation" not in anno:
        return False
    
    segm = anno["segmentation"]
    
    # 2. 如果是多边形格式 (list)
    if isinstance(segm, list):
        if len(segm) == 0:
            return False
        # 检查每一个多边形是否有至少 3 个点 (6 个坐标值)
        for poly in segm:
            if len(poly) < 6: # x1,y1,x2,y2,x3,y3
                return False
    # 3. 如果是 RLE 格式 (dict)
    elif isinstance(segm, dict):
        if "counts" not in segm or "size" not in segm:
            return False
    else:
        return False
        
    return True



def main(args):
    #  Data loading and filtering logic
    json_file = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train_fixed_generalized.json"
    image_root = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train"

    full_dataset_dicts = load_coco_json(json_file, image_root, dataset_name="full_data")

    full_meta = MetadataCatalog.get("full_data")
    id_map = full_meta.thing_dataset_id_to_contiguous_id  # key：{1:0, 2:1, ..., 8:7}
    thing_classes = full_meta.thing_classes   
    
    
    filtered_dicts = []
    for d in full_dataset_dicts:
        # Keep only valid annotations (prevent training from crashing with area 0 or invalid coordinates)
        valid_annos = [anno for anno in d["annotations"] if is_valid_coco_anno(anno)]
        
        # If there are valid annotations in the image, we keep the image
        if len(valid_annos) > 0:
            d["annotations"] = valid_annos
            filtered_dicts.append(d)

    print(f"数据清洗结果 - 原始图片: {len(full_dataset_dicts)}, 过滤后有效图片: {len(filtered_dicts)}")
    
    #  train : val = 8:2
    random.seed(42)
    random.shuffle(filtered_dicts)
    split = int(0.8 * len(filtered_dicts))
    
    train_dicts = filtered_dicts[:split]
    val_dicts = filtered_dicts[split:]
    
    
    # my_class_names = ["cellpose", "DeepBacs" ,"DSB" ,"HEK293T", "livecell", "MoNuSAC" ,"Tissue","TNBC"]
    my_class_names = ["nucleus", "cell"]
    
    if "coco_my_train" not in DatasetCatalog.list():
        DatasetCatalog.register("coco_my_train", lambda: train_dicts)
        # thing_classes
        MetadataCatalog.get("coco_my_train").set(
            thing_classes=my_class_names, 
            json_file=json_file, 
            image_root=image_root, 
            evaluator_type="coco"
        )
    
    if "coco_my_val" not in DatasetCatalog.list():
        DatasetCatalog.register("coco_my_val", lambda: val_dicts)
        MetadataCatalog.get("coco_my_val").set(
            thing_classes=my_class_names, 
            json_file=json_file, 
            image_root=image_root, 
            evaluator_type="coco"
        )
    
    MetadataCatalog.get("coco_my_train").set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=id_map,   # <- 关键一行
    )

    MetadataCatalog.get("coco_my_val").set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=id_map,   # <- 关键一行
    )


    print_dataset_stats("coco_my_train")
    u = sorted({a["category_id"] for d in DatasetCatalog.get("coco_my_train") for a in d["annotations"]})
    print("unique category_id in dicts:", u)
    
    # init
    cfg = setup(args)

    # valid
    if args.eval_only:
        # Instantiating a model
        model = Trainer.build_model(cfg)
        
        # param check
        print("--- Trainable Parameters ---")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
        
        # load weight
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        
        # 基础测试
        res = Trainer.test(cfg, model)
        
        # 【功能增加】支持 TTA (Test Time Augmentation)
        if cfg.TEST.AUG.ENABLED:
            print("--- 正在运行 TTA 推理增强 ---")
            res.update(Trainer.test_with_TTA(cfg, model))
            
        # 主进程打印最终验证结果
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # 如果开启了 TTA，将其注册到训练钩子中
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
        
    return trainer.train()

# 在 main 函数注册完数据后加入以下代码：
def print_dataset_stats(dataset_name):
    # 获取数据集字典
    dataset_dicts = DatasetCatalog.get(dataset_name)
    # 获取元数据（类别名）
    metadata = MetadataCatalog.get(dataset_name)
    classes = metadata.thing_classes
    
    from collections import Counter
    counts = Counter()
    for d in dataset_dicts:
        for anno in d["annotations"]:
            counts[anno["category_id"]] += 1
            
    print(f"\n数据集 [{dataset_name}] 统计结果:")
    print("-" * 30)
    for i, class_name in enumerate(classes):
        print(f"类别 ID: {i} | 名称: {class_name:15} | 数量: {counts[i]}")
    print("-" * 30)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )