# printf_info.py
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import model_info

yaml_path = r'E:\yolo-rknn\ultralytics\cfg\models\12\yolo12n_npu_h.yaml'

# 构建模型
model = DetectionModel(cfg=yaml_path, nc=24)
model.eval()

# ── 1. 官方 model_info（params / GFLOPs）─
model_info(model, imgsz=640, verbose=True)

# ── 2. 逐层参数统计 ──
print("\n" + "="*70)
print(f"{'Layer':<5} {'Type':<40} {'Params':>10} {'Trainable':>10}")
print("="*70)
total, trainable = 0, 0
for i, (name, param) in enumerate(model.named_parameters()):
    total     += param.numel()
    trainable += param.numel() if param.requires_grad else 0
for i, layer in enumerate(model.model):
    lp = sum(p.numel() for p in layer.parameters())
    print(f"{i:<5} {str(type(layer).__name__):<40} {lp:>10,}")
print("="*70)
print(f"{'Total params':<46} {total:>10,}")
print(f"{'Trainable params':<46} {trainable:>10,}")
print(f"{'Non-trainable params':<46} {total-trainable:>10,}")

# # ── 3. 手动计算 FLOPs（用 thop）──
# print("\n" + "="*70)
# try:
#     from thop import profile, clever_format
#     dummy = torch.zeros(1, 3, 640, 640)
#     macs, params = profile(model, inputs=(dummy,), verbose=False)
#     macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
#     print(f"  Input size   : 1 × 3 × 640 × 640")
#     print(f"  MACs (GFLOPs): {macs_fmt}  ({float(macs)/1e9:.2f} G)")
#     print(f"  Params       : {params_fmt}  ({float(params)/1e6:.2f} M)")
# except ImportError:
#     print("  thop 未安装，运行:  pip install thop")
#     # 备用：torchinfo
#     try:
#         from torchinfo import summary
#         summary(model, input_size=(1, 3, 640, 640),
#                 col_names=["input_size","output_size","num_params","mult_adds"],
#                 depth=3)
#     except ImportError:
#         print("  torchinfo 也未安装，运行:  pip install torchinfo")
#
# # ── 4. 推理延迟测试 ──
# print("\n" + "="*70)
# print("  延迟测试 (CPU)...")
# dummy = torch.zeros(1, 3, 640, 640)
# # 预热
# with torch.no_grad():
#     for _ in range(5):
#         model(dummy)
#
# import time
# runs = 50
# with torch.no_grad():
#     t0 = time.perf_counter()
#     for _ in range(runs):
#         model(dummy)
#     elapsed = (time.perf_counter() - t0) / runs * 1000
# print(f"  平均推理延迟 : {elapsed:.2f} ms/frame  (CPU, {runs}次均值)")
# print(f"  理论帧率     : {1000/elapsed:.1f} FPS")
#
# # ── 5. 模型内存占用 ──
# print("\n" + "="*70)
# param_mb  = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
# buffer_mb = sum(b.numel() * b.element_size() for b in model.buffers())    / 1024**2
# print(f"  参数内存 (FP32) : {param_mb:.2f} MB")
# print(f"  Buffer内存      : {buffer_mb:.2f} MB")
# print(f"  模型总内存估计  : {param_mb + buffer_mb:.2f} MB")
# print("="*70)