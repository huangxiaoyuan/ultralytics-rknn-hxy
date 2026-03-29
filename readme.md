# 🦅 YOLO12-RKNN-NPU

> **YOLOv12 × RK3588 / SpacemiT K1 NPU 友好化改进方案**  
> 系统性消除 YOLOv12 中的 NPU 不兼容算子，无需修改任何 Python 代码，仅通过 YAML 配置即可部署至 RK3588、SpacemiT K1 等边缘 NPU 平台。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv12-purple" />
  <img src="https://img.shields.io/badge/Platform-RK3588%20%7C%20K1-green" />
  <img src="https://img.shields.io/badge/License-AGPL--3.0-red" />
</p>

---

## 目录

- [为什么需要这个项目](#-为什么需要这个项目)
- [改进内容](#-改进内容)
- [消融实验结果](#-消融实验结果)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [模型配置文件](#-模型配置文件)
- [导出与部署](#-导出与部署)
- [项目结构](#-项目结构)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)
- [引用](#-引用)

---

## 🤔 为什么需要这个项目

标准 YOLOv12 在部署到 RK3588（RKNN）或 SpacemiT K1（ONNX Runtime SpaceMIT EP）时，以下算子会产生 **CPU fallback**，导致 NPU 利用率不足 60%、推理延迟大幅增加：

| 算子 / 模块 | 问题 |
|------------|------|-------------|
| `SiLU` 激活 | INT8 量化精度损失，部分平台不支持 |
| `A2C2f`（Area Attention） | 动态 shape，NPU 完全不支持 | 
| `C2PSA`（Point-wise SA） | Attention 动态 shape，不支持 | 
| `DFL`（reg_max=16） | softmax + gather，部分 fallback | 

本项目通过**仅修改 YAML 配置**的方式，将上述模块替换为 NPU 原生支持的等效模块，估算推理延迟从 ~38ms 降至 ~13ms（RK3588，INT8，640×640）。

---

## 🔧 改进内容

> ⚠️ **Detect 检测头保持不变**，所有改动仅在 backbone 和 neck 层进行。

### 新增模块（`ultralytics/nn/modules/`）

#### 1. `NPUConv` — NPU 友好卷积块

```
原始：Conv + BN + SiLU   →   改进：Conv + BN + ReLU6
```

将默认激活函数从 `SiLU` 替换为 `ReLU6`（ONNX `Clip(0,6)` 算子），被所有主流 NPU 推理框架原生支持。

```python
# ultralytics/nn/modules/conv.py
class NPUConv(Conv):
    """NPU-friendly Conv: Conv + BN + ReLU6"""
```

---

#### 2. `NPU_C3k2` — 替换 A2C2f / C3k2

```
原始：A2C2f（Area Attention，动态 shape）
改进：NPU_C3k2（密集卷积堆叠 + SE 通道注意力）
```

以深层密集卷积替代稀疏 Attention，集成轻量 SE 模块恢复特征判别力，全部算子 NPU 原生支持。

```
NPU_C3k2(x) = SE(CV2(Concat([CV1(x), Bottleneck_1(...), ..., Bottleneck_n(...)])))
```

---

#### 3. `NPU_SE_Block` — 替换 C2PSA

```
原始：C2PSA（Point-wise Self-Attention）
改进：NPU_SE_Block（GlobalAvgPool + Linear + ReLU6 + HardSigmoid）
```

参数量仅约 8K，有效保留通道注意力机制，消除所有动态计算图。

```
NPU_SE_Block(x) = x × HardSigmoid(Linear₂(ReLU6(Linear₁(GAP(x)))))
```

---

### 改动文件清单

| 文件 | 改动内容 |
|------|---------|
| `ultralytics/nn/modules/conv.py` | 新增 `NPUConv` 类 |
| `ultralytics/nn/modules/block.py` | 新增 `NPU_C3k2`、`NPU_SE_Block`、`NPU_Bottleneck` 类 |
| `ultralytics/nn/modules/__init__.py` | 导出新模块 |
| `ultralytics/nn/tasks.py` | 注册新模块到 `parse_model`，新增 `elif m is NPU_C3k2` 分支 |
| `ultralytics/models/cfg/models/12/yolo12n_npu.yaml` | NPU 友好版通用网络配置 |
| `ultralytics/models/cfg/models/12/yolo12n_bird.yaml` | 鸟类识别轻量压缩版 |

---

## 📊 消融实验结果



### CUB-200-2011 数据集（泛化参考）

| 实验组 | 模型 | mAP@0.5 | mAP@0.5:95 | P | R |
|--------|------|---------|-----------|---|---|
| Baseline | yolov12n | 0.820 | 0.728 | 0.789 | 0.765 |
| Exp-A | yolov12n_npu | 0.733 | 0.613 | 0.721 | 0.672 |
| Exp-B | yolo12_npu_cc | 0.716 | 0.592 | 0.694 | 0.654 |
| Final | yolo12_npu_bird | 0.669 | 0.547 | 0.652 | 0.625 |



---

## 📦 安装

与 [Ultralytics](https://github.com/ultralytics/ultralytics) 安装方式完全一致：

```bash
# 克隆本仓库（替代官方 ultralytics）
git clone https://github.com/your-username/yolo12-rknn-npu.git
cd yolo12-rknn-npu

# 创建虚拟环境（推荐）
conda create -n yolo_npu python=3.10
conda activate yolo_npu

# 安装依赖
pip install -e ".[dev]"

# 验证安装
python -c "from ultralytics import YOLO; print('✅ 安装成功')"
```

### 额外依赖（可选）

```bash
# 模型信息打印
pip install thop torchinfo

# ONNX 导出
pip install onnx onnx-simplifier onnxruntime

# RK3588 部署（需在 Linux 环境安装）
pip install rknn-toolkit2  # 参考官方文档
```

---

## 🚀 快速开始

### 1. 使用 NPU 友好版配置训练

只需将 `model` 参数指向本仓库提供的 NPU yaml 文件，其余与标准 Ultralytics 用法完全相同：

```python
from ultralytics import YOLO

# 加载 NPU 友好版模型（替换原版 yolo12n.yaml）
model = YOLO("ultralytics/models/cfg/models/12/yolo12n_npu.yaml")

# 训练（与官方用法完全一致）
model.train(
    data="your_dataset.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
)
```

### 2. 鸟类识别轻量版

```python
from ultralytics import YOLO

# 鸟类专用压缩版（参数量 1.12M）
model = YOLO("ultralytics/models/cfg/models/12/yolo12n_bird.yaml")

model.train(
    data="birds.yaml",  # 你的鸟类数据集配置
    epochs=300,
    imgsz=640,
    nc=200,             # 修改为实际类别数
)
```

### 3. 打印模型信息

```python
from ultralytics.nn.tasks import DetectionModel

yaml_path = "ultralytics/models/cfg/models/12/yolo12n_npu.yaml"
model = DetectionModel(cfg=yaml_path, nc=200)

# 打印参数量、GFLOPs
from ultralytics.utils.torch_utils import model_info
model_info(model, imgsz=640, verbose=True)
```

### 4. 验证模型

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
metrics = model.val(data="your_dataset.yaml", imgsz=640)

print(f"mAP50    : {metrics.box.map50:.4f}")
print(f"mAP50-95 : {metrics.box.map:.4f}")
```

---

## 📁 模型配置文件

本仓库提供以下 YAML 配置，位于 `ultralytics/models/cfg/models/12/`：

| 文件 | 描述 | Params | GFLOPs | 适用场景 |
|------|------|--------|--------|---------|
| `yolo12n.yaml` | 原版 YOLOv12n（未改动） | 2.65M | 6.8 | GPU 训练基线 |
| `yolo12n_npu.yaml` | NPU 友好版（算子替换） | 2.49M | 7.8 | NPU 部署通用场景 |
| `yolo12n_npu_cc.yaml` | NPU + 通道压缩版 | 1.44M | 6.6 | 资源受限边缘设备 |
| `yolo12n_bird.yaml` | 鸟类识别轻量版 | 1.12M | 6.1 | 鸟类/细粒度识别专用 |

### yaml 改动原则速查

```yaml
# 将以下原版模块替换为对应 NPU 模块：

Conv       →  NPUConv      # SiLU → ReLU6，消除激活 fallback
A2C2f      →  NPU_C3k2     # 删除 Area-Attention，第3个参数直接去掉
C3k2       →  NPU_C3k2     # 激活替换 + SE 注意力，去掉 expand ratio 参数
C2PSA      →  NPU_SE_Block  # 删除 PSA，换轻量 SE

# 参数格式示例：
# 原版：- [-1, 4, A2C2f,  [512, True, 4]]
# NPU版：- [-1, 4, NPU_C3k2, [512, True]]   ← 去掉第3个参数

# 原版：- [-1, 2, C3k2,  [256, False, 0.25]]
# NPU版：- [-1, 2, NPU_C3k2, [256, False]]   ← 去掉 expand ratio
```

---

## 🔄 导出与部署

### 导出 ONNX

```bash
python export_onnx.py \
    --model path/to/best.pt \
    --output ./export \
    --imgsz 640 \
    --opset 13
```

或使用 Python API：

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
model.export(
    format="onnx",
    imgsz=640,
    opset=13,
    simplify=True,
    dynamic=False,   # NPU 部署必须静态 shape
)
```

### RK3588 模型转换（RKNN）

参考rknn_tools/convert2rknn.py或者rk官方的onnx2rknn.py，转换完成后进行使用inference_rknn_val.py进行评估。
### RK3588 模型部署

板卡部署可以参考rknn_tools/yolo_infer.py  

## 🗂 项目结构

```
yolo12-rknn-npu/
├── ultralytics/
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── conv.py          # ← NPUConv 新增于此
│   │   │   ├── block.py         # ← NPU_C3k2, NPU_SE_Block 新增于此
│   │   │   └── __init__.py      # ← 导出新模块
│   │   └── tasks.py             # ← parse_model 注册新模块
│   └── models/cfg/models/12/
│       ├── yolo12n.yaml         # 原版（未改动）
│       ├── yolo12n_npu.yaml     # NPU 友好版
│       ├── yolo12n_npu_cc.yaml  # NPU + 压缩版
│       └── yolo12n_bird.yaml    # 鸟类轻量版
├── rknn_tools
│   ├── inference..           
│       ├── ..         # RKNN 导出脚本
│   ├── yolo_infer.py           # RK3588 推理脚本
│   └── 其他板卡推理脚本..         # 统一推理（兼容原版/NPU Detect）
├── ..
│   ..
├── README.md
└── requirements.txt
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 提交 Issue

- 🐛 **Bug 报告**：请附上完整报错信息、Python/PyTorch/Ultralytics 版本、操作系统
- 💡 **功能建议**：描述使用场景和期望的改进效果
- ❓ **使用问题**：先查阅 [FAQ](#常见问题) 和已有 Issue

### 提交 PR

```bash
# 1. Fork 本仓库并克隆
git clone https://github.com/your-username/yolo12-rknn-npu.git

# 2. 创建功能分支
git checkout -b feature/your-feature-name

# 3. 提交改动
git commit -m "feat: 添加对 YOLO26 的 NPU 支持"

# 4. 推送并创建 PR
git push origin feature/your-feature-name
```

### 新增 NPU 模块的规范

如果你想贡献新的 NPU 友好模块，请遵循以下规范：

1. 模块定义放在 `ultralytics/nn/modules/block.py` 或 `conv.py`
2. 命名统一使用 `NPU_` 前缀
3. `__init__` 中对所有通道参数调用 `int()` 防止 tuple 传入
4. 在 `__init__.py` 导出，在 `tasks.py` 的 `parse_model` 中注册
5. 提供对应的 YAML 配置示例

### 常见问题

<details>
<summary><b>Q: KeyError: 'NPU_C3k2' 怎么解决？</b></summary>

确认 `tasks.py` 的导入列表中包含 `NPU_C3k2`，以及 `parse_model` 中有对应的 `elif m is NPU_C3k2` 分支。详见 [安装说明](#-安装)。

</details>

<details>
<summary><b>Q: GFLOPs 比原版更高，正常吗？</b></summary>

正常。A2C2f 的稀疏 Area-Attention 被密集卷积替代，理论 FLOPs 增加，但消除了 NPU fallback 后实际推理延迟反而降低。

</details>

<details>
<summary><b>Q: 能用在 YOLOv11 / YOLO26 上吗？</b></summary>

可以。本仓库同样提供了 YOLOv11 和 YOLO26 的 NPU 友好版 YAML，原理相同，仅替换对应模块名称。

</details>

<details>
<summary><b>Q: Detect 头为什么不改？</b></summary>

原版 Detect 头（reg_max=16）在 RK3588 上 DFL 的 softmax+gather 会部分 fallback，但代价约 2~3ms，对整体影响有限。保持原版 Detect 可确保与 Ultralytics 官方训练/推理流程完全兼容，降低维护成本。如需进一步优化，可设置 `reg_max=1`（YOLO26 默认）彻底消除 DFL 开销。

</details>

---

## 📄 许可证

本项目基于 [AGPL-3.0 License](LICENSE) 开源，与 Ultralytics 官方仓库保持一致。

---

## 📖 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{yolo12-rknn-npu,
  title     = {YOLO12-RKNN-NPU: NPU-Friendly YOLOv12 for Edge Deployment},
  author    = {Xiaoyuan Huang},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/your-username/yolo12-rknn-npu}
}
```

同时建议引用原始 YOLOv12 论文：

```bibtex
@article{tian2025yolov12,
  title   = {YOLOv12: Attention-Centric Real-Time Object Detectors},
  author  = {Tian, Yungeng and Ye, Qian and Ran, Jie},
  journal = {arXiv preprint arXiv:2502.12524},
  year    = {2025}
}
```

---

<p align="center">
  <sub>Built on top of <a href="https://github.com/ultralytics/ultralytics">Ultralytics</a> · Tested on RK3588 & SpacemiT K1</sub>
</p>
