# evaluate.py
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ================================================================
# 配置区域 - 根据实际情况修改
# ================================================================
MODEL_PATH = "/data/rknn/runs/detect/train2/weights/best.pt"
DATA_CONFIG = "ultralytics/cfg/datasets/bird_cub200_server.yaml"  # 数据集配置文件
IMG_SIZE = 640
DEVICE = "0"  # 'cpu' 或 '0'(GPU)
CONF_THRES = 0.25
IOU_THRES = 0.45
# ================================================================


def evaluate_metrics(model):
    """① 核心指标评估：mAP / Precision / Recall."""
    print("\n" + "=" * 60)
    print("  ① 核心指标评估")
    print("=" * 60)

    # results = model.val(
    #     data=DATA_CONFIG,
    #     imgsz=IMG_SIZE,
    #     conf=CONF_THRES,
    #     iou=IOU_THRES,
    #     device=DEVICE,
    #     verbose=True,
    #     plots=True,           # 自动保存混淆矩阵、PR曲线
    #     save_json=True,       # 保存COCO格式json
    # )
    results = model.val(data=DATA_CONFIG)
    mp = results.box.mp  # mean Precision
    mr = results.box.mr  # mean Recall
    map50 = results.box.map50  # mAP@0.5
    map5095 = results.box.map  # mAP@0.5:0.95

    print(f"\n  {'指标':<20} {'值':>10}")
    print(f"  {'-' * 32}")
    print(f"  {'Precision':<20} {mp:>10.4f}")
    print(f"  {'Recall':<20} {mr:>10.4f}")
    print(f"  {'mAP@0.5':<20} {map50:>10.4f}")
    print(f"  {'mAP@0.5:0.95':<20} {map5095:>10.4f}")
    print(f"  {'F1-score':<20} {2 * mp * mr / (mp + mr + 1e-8):>10.4f}")

    return results


def evaluate_per_class(results, top_n=20):
    """② 逐类别指标（取Top-N最差类别，便于针对性改进）."""
    print("\n" + "=" * 60)
    print("  ② 逐类别指标（Top-20最差 mAP@0.5）")
    print("=" * 60)

    names = results.names  # {0:'sparrow', 1:'eagle',...}
    ap50 = results.box.ap50  # 每类 AP@0.5，shape=(nc,)
    ap = results.box.ap  # 每类 AP@0.5:0.95
    p_cls = results.box.p  # 每类 Precision
    r_cls = results.box.r  # 每类 Recall

    # 按 mAP50 升序排列（最差在前）
    order = np.argsort(ap50)

    print(f"\n  {'类别':<25} {'P':>7} {'R':>7} {'AP50':>8} {'AP50:95':>10}")
    print(f"  {'-' * 60}")
    for i in order[:top_n]:
        name = names[i] if isinstance(names, dict) else str(i)
        print(f"  {name:<25} {p_cls[i]:>7.3f} {r_cls[i]:>7.3f} {ap50[i]:>8.3f} {ap[i]:>10.3f}")

    # 同时打印最好的Top-N
    print(f"\n  {'--- Top-20 最好类别 ---':^60}")
    print(f"  {'类别':<25} {'P':>7} {'R':>7} {'AP50':>8} {'AP50:95':>10}")
    print(f"  {'-' * 60}")
    for i in order[-top_n:][::-1]:
        name = names[i] if isinstance(names, dict) else str(i)
        print(f"  {name:<25} {p_cls[i]:>7.3f} {r_cls[i]:>7.3f} {ap50[i]:>8.3f} {ap[i]:>10.3f}")

    return ap50, names


def evaluate_speed(model):
    """③ 推理速度测试."""
    print("\n" + "=" * 60)
    print("  ③ 推理速度测试")
    print("=" * 60)
    import time

    import torch

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    model.model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            model.model(dummy)

    # 计时
    runs = 100
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(runs):
            model.model(dummy)
        elapsed = (time.perf_counter() - t0) / runs * 1000

    print(f"\n  输入尺寸  : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  平均延迟  : {elapsed:.2f} ms/frame")
    print(f"  理论帧率  : {1000 / elapsed:.1f} FPS")
    print(f"  设备      : {DEVICE}")


def evaluate_visual(model, num_images=8):
    """④ 可视化检测结果（随机抽样val集图片）."""
    print("\n" + "=" * 60)
    print("  ④ 可视化检测结果")
    print("=" * 60)

    import random

    import yaml

    with open(DATA_CONFIG) as f:
        data_cfg = yaml.safe_load(f)

    val_path = data_cfg.get("val", "")
    if not os.path.isabs(val_path):
        val_path = os.path.join(os.path.dirname(DATA_CONFIG), val_path)

    # 收集图片
    img_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_files.extend(Path(val_path).rglob(ext))
    if not img_files:
        print("  未找到验证集图片，跳过可视化")
        return

    samples = random.sample(img_files, min(num_images, len(img_files)))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle("YOLO Bird Detection - Validation Samples", fontsize=14)

    for idx, (img_path, ax) in enumerate(zip(samples, axes)):
        results = model.predict(str(img_path), conf=CONF_THRES, iou=IOU_THRES, device=DEVICE, verbose=False)
        # 画图
        annotated = results[0].plot()  # BGR numpy
        annotated = annotated[:, :, ::-1]  # BGR→RGB
        ax.imshow(annotated)
        ax.axis("off")

        # 统计检测数量
        n_det = len(results[0].boxes)
        ax.set_title(f"{img_path.stem[:20]} | {n_det} det", fontsize=8)

    plt.tight_layout()
    save_path = r"E:\yolo-rknn\runs\eval_visual.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  可视化结果已保存: {save_path}")
    plt.show()


def evaluate_confusion(results):
    """⑤ 混淆矩阵分析（自动由val()生成，这里打印数值摘要）."""
    print("\n" + "=" * 60)
    print("  ⑤ 混淆矩阵摘要")
    print("=" * 60)

    try:
        cm = results.confusion_matrix.matrix  # shape: (nc+1, nc+1)
        nc = cm.shape[0] - 1

        # 对角线准确率
        diag = np.diag(cm[:nc, :nc])
        total = cm[:nc, :nc].sum(axis=1) + 1e-8
        acc_per_class = diag / total

        print(f"\n  总类别数        : {nc}")
        print(f"  平均分类准确率  : {acc_per_class.mean():.4f}")
        print(f"  最高准确率类别  : {acc_per_class.max():.4f}")
        print(f"  最低准确率类别  : {acc_per_class.min():.4f}")
        print(f"  准确率<0.5的类别: {(acc_per_class < 0.5).sum()} 个")

        # FP/FN分析
        fp = cm[:nc, nc].sum()  # 背景被误检为目标
        fn = cm[nc, :nc].sum()  # 目标被漏检为背景
        print(f"\n  误检(FP)总数    : {int(fp)}")
        print(f"  漏检(FN)总数    : {int(fn)}")

    except Exception as e:
        print(f"  混淆矩阵读取失败: {e}")
        print("  请查看 runs/detect/val/ 目录下的 confusion_matrix.png")


def save_report(results, ap50, names):
    """⑥ 保存评估报告到txt."""
    report_path = r"E:\yolo-rknn\runs\eval_report.txt"
    mp = results.box.mp
    mr = results.box.mr
    map50 = results.box.map50
    map5095 = results.box.map

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("YOLO Bird Detection 评估报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型路径  : {MODEL_PATH}\n")
        f.write(f"数据配置  : {DATA_CONFIG}\n")
        f.write(f"输入尺寸  : {IMG_SIZE}\n\n")
        f.write(f"Precision : {mp:.4f}\n")
        f.write(f"Recall    : {mr:.4f}\n")
        f.write(f"mAP@0.5   : {map50:.4f}\n")
        f.write(f"mAP@0.5:95: {map5095:.4f}\n")
        f.write(f"F1-score  : {2 * mp * mr / (mp + mr + 1e-8):.4f}\n\n")
        f.write("逐类别AP@0.5（升序）\n")
        f.write("-" * 40 + "\n")
        for i in np.argsort(ap50):
            name = names[i] if isinstance(names, dict) else str(i)
            f.write(f"{name:<30} {ap50[i]:.4f}\n")

    print(f"\n  评估报告已保存: {report_path}")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    print("加载模型...")
    model = YOLO(MODEL_PATH)

    results = evaluate_metrics(model)  # ① 核心指标
    ap50, names = evaluate_per_class(results)  # ② 逐类别
    # evaluate_speed(model)                       # ③ 速度
    # evaluate_visual(model)                      # ④ 可视化
    evaluate_confusion(results)  # ⑤ 混淆矩阵
    save_report(results, ap50, names)  # ⑥ 保存报告

    print("\n✅ 评估完成")
