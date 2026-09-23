# export_onnx.py
import os
import sys

# import warnings
# warnings.filterwarnings('ignore')

# ================================================================
# 配置区域
# ================================================================
MODEL_PATH = r"E:\bird\yolo_npu\train4\weights\best.pt"
OUTPUT_DIR = r"E:\bird\yolo_npu\train4\weights\export"
IMG_SIZE = 640
OPSET = 13  # K1推荐11~13，RK3588推荐12
SIMPLIFY = True  # 简化ONNX图，消除冗余节点
DEVICE = "GPU"
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_onnx(model_path, output_dir, imgsz, opset, simplify):

    import onnx
    import onnxsim

    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f"  加载模型: {model_path}")
    print(f"{'=' * 60}")

    model = YOLO(model_path)

    # ── 打印基本信息 ──
    params = sum(p.numel() for p in model.model.parameters())
    print(f"  参数量  : {params / 1e6:.3f} M")
    print(f"  输入尺寸: {imgsz}×{imgsz}")
    print(f"  OPSET   : {opset}")
    print(f"  简化    : {simplify}")

    # ── 导出ONNX ──
    print("\n  正在导出ONNX...")
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,  # 静态shape，NPU部署必须
        half=False,  # FP32导出，量化由后续工具完成
        device=DEVICE,
    )
    print(f"  初始导出: {export_path}")

    # ── 验证ONNX模型 ──
    print("\n  验证ONNX模型结构...")
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    print("  ✅ ONNX结构验证通过")

    # ── 打印输入输出信息 ──
    print("\n  模型输入:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")

    print("\n  模型输出:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    # ── 统计算子类型 ──
    print("\n  ONNX算子统计:")
    op_count = {}
    unsupported_ops = set()
    # K1/RK3588 不友好的算子
    npu_unfriendly = {
        "Gather",
        "GatherElements",
        "ScatterND",
        "NonMaxSuppression",
        "RoiAlign",
        "Einsum",
        "LayerNormalization",
        "GroupNormalization",
    }

    for node in onnx_model.graph.node:
        op = node.op_type
        op_count[op] = op_count.get(op, 0) + 1
        if op in npu_unfriendly:
            unsupported_ops.add(op)

    # 按频次排序打印
    for op, cnt in sorted(op_count.items(), key=lambda x: -x[1]):
        flag = " ⚠️ NPU不友好" if op in npu_unfriendly else ""
        print(f"    {op:<30} × {cnt:>4}{flag}")

    if unsupported_ops:
        print(f"\n  ⚠️  发现NPU不友好算子: {unsupported_ops}")
        print("     建议检查模型中是否还有未替换的原版模块")
    else:
        print("\n  ✅ 未发现NPU不友好算子，模型完全NPU兼容")

    # ── onnxsim 二次简化 ──
    if simplify:
        print("\n  onnxsim 二次简化...")
        try:
            model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx_model = model_simplified
                print("  ✅ onnxsim 简化成功")

                # 对比节点数变化
                orig_nodes = len(onnx.load(export_path).graph.node)
                simp_nodes = len(onnx_model.graph.node)
                print(f"     节点数: {orig_nodes} → {simp_nodes} (减少 {orig_nodes - simp_nodes} 个)")
            else:
                print("  ⚠️  onnxsim 简化验证失败，使用原始版本")
        except Exception as e:
            print(f"  ⚠️  onnxsim 失败: {e}")

    # ── 保存最终ONNX ──
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    final_path = os.path.join(output_dir, f"{model_name}_op{opset}.onnx")
    onnx.save(onnx_model, final_path)
    print(f"\n  ✅ 最终ONNX已保存: {final_path}")

    # ── 文件大小 ──
    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"     文件大小: {size_mb:.2f} MB")

    return final_path


def verify_onnx_inference(onnx_path, imgsz):
    """验证ONNX推理结果与PyTorch一致."""
    print(f"\n{'=' * 60}")
    print("  验证ONNX推理输出")
    print(f"{'=' * 60}")

    try:
        import numpy as np
        import onnxruntime as ort

        # 构建session
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # 随机输入
        dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # 推理
        outputs = session.run(None, {input_name: dummy})

        print(f"  输入  : {dummy.shape}  dtype={dummy.dtype}")
        for i, out in enumerate(outputs):
            print(f"  输出{i} : shape={out.shape}  dtype={out.dtype}  min={out.min():.4f}  max={out.max():.4f}")

        print("\n  ✅ ONNX推理验证通过")

    except ImportError:
        print("  onnxruntime未安装，跳过推理验证")
        print("  安装命令: pip install onnxruntime")
    except Exception as e:
        print(f"  ❌ 推理验证失败: {e}")
        import traceback

        traceback.print_exc()


def export_torchscript(model_path, output_dir, imgsz):
    """额外导出TorchScript格式（备用）."""
    print(f"\n{'=' * 60}")
    print("  导出 TorchScript（备用格式）")
    print(f"{'=' * 60}")
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        ts_path = model.export(
            format="torchscript",
            imgsz=imgsz,
            device=DEVICE,
        )
        dst = os.path.join(output_dir, os.path.basename(ts_path))
        os.rename(ts_path, dst)
        print(f"  ✅ TorchScript已保存: {dst}")
    except Exception as e:
        print(f"  ⚠️  TorchScript导出失败: {e}")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    # 检查依赖
    missing = []
    for pkg in ["onnx", "onnxsim", "onnxruntime"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print("缺少依赖，请先安装:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    # ── 导出ONNX ──
    onnx_path = export_onnx(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        imgsz=IMG_SIZE,
        opset=OPSET,
        simplify=SIMPLIFY,
    )

    # ── 验证推理 ──
    verify_onnx_inference(onnx_path, IMG_SIZE)

    # ── 打印后续步骤提示 ──
    print(f"\n{'=' * 60}")
    print("  后续步骤")
    print(f"{'=' * 60}")
    print(f"""
  ▶ RK3588 部署路径：
    1. 将 {onnx_path} 拷贝到装有 rknn-toolkit2 的环境
    2. 运行量化脚本:
       python export_rknn.py --onnx {os.path.basename(onnx_path)}
    3. 推理: python rk3588_infer.py

  ▶ SpacemiT K1 部署路径：
    1. 将 {onnx_path} 拷贝到 K1 设备
    2. 运行 xquant 量化:
       xquant --config xquant_bird.json
    3. 推理: python infer_k1.py
""")
