import onnx, sys
m = onnx.load("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/Export/model.onnx")
ops = sorted({n.op_type for n in m.graph.node})
print("OPS:", ops)
print("Has If:", "If" in ops)
print("Outputs:", [o.name for o in m.graph.output])
