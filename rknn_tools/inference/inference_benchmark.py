import subprocess
import os
import re
from datetime import datetime

# ================= 配置区域 =================
BENCHMARK_SCRIPT = "inference_time_val.py"
SAVE_RESULT_FILE = "thesis_benchmark_full_report.txt"

# 按照您的要求列出的 8 个模型 (请确保路径正确)
MODELS_TO_TEST = [
    {"path": "bird_rknn_model/320/best_yolov12n_320_fp.rknn", "name": "best_yolov12n_320_fp"},
    {"path": "bird_rknn_model/320/best_yolov12_320_op13.rknn", "name": "best_yolov12_320_op13"},
    {"path": "bird_rknn_model/320/best_npu_cc_320_op13_fp.rknn", "name": "best_npu_cc_320_op13_fp"},
    {"path": "bird_rknn_model/320/best_npu_cc_320_op13.rknn", "name": "best_npu_cc_320_op13"},
    {"path": "bird_rknn_model/320/best_npu_320_op13_fp.rknn", "name": "best_npu_320_op13_fp"},
    {"path": "bird_rknn_model/320/best_npu_320_op13.rknn", "name": "best_npu_320_op13"},
    {"path": "bird_rknn_model/320/best_bird_320_op13_fp.rknn", "name": "best_bird_320_op13_fp"},
    {"path": "bird_rknn_model/320/best_bird_320_op13.rknn", "name": "best_bird_320_op13"},
]

TEST_IMAGE = "Hydrophasianus_chirurgus_172.jpg"
BENCH_FRAMES = 100
WARMUP_FRAMES = 10


# ================= 核心解析逻辑 =================

def extract_table_row(row_name, output):
    """
    使用正则精准提取表格中某一行的数据:
    阶段名 | Min | Max | Mean | P50 | P90 | P99 | Std
    """
    # 匹配模式: 阶段名 + 分隔符 + 7个浮点数
    pattern = rf"{row_name}\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)"
    match = re.search(pattern, output)
    if match:
        return {
            "min": match.group(1), "max": match.group(2), "mean": match.group(3),
            "p50": match.group(4), "p90": match.group(5), "p99": match.group(6), "std": match.group(7)
        }
    return None


def format_model_report(model_name, output):
    """提取该模型的所有阶段指标并格式化为表格字符串"""
    npu = extract_table_row("NPU Inference", output)
    post = extract_table_row("Post-Process", output)
    total = extract_table_row("Total Pipeline", output)

    # 提取 FPS 指标
    fps_match = re.search(r"平均吞吐量 \(Mean FPS\):\s+([\d.]+)", output)
    fps = fps_match.group(1) if fps_match else "N/A"

    if not npu or not post or not total:
        return f"\n[!] 模型 {model_name} 数据提取失败，请检查压测脚本输出格式。\n"

    report = f"\n### 模型名称: {model_name}\n"
    report += f"平均吞吐量: {fps} FPS\n"
    report += "阶段 (ms)       |   Min |   Max |  Mean |   P50 |   P90 |   P99 |  Std\n"
    report += "-" * 75 + "\n"

    for label, d in [("NPU Inference ", npu), ("Post-Process  ", post), ("Total Pipeline ", total)]:
        report += f"{label:<15} | {d['min']:>5} | {d['max']:>5} | {d['mean']:>5} | {d['p50']:>5} | {d['p90']:>5} | {d['p99']:>5} | {d['std']:>4}\n"

    report += "-" * 75 + "\n"
    return report


# ================= 主流程 =================

def run_main():
    final_content = f"YOLOv12 RK3588 性能测试全指标报告\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    final_content += f"压测参数: 帧数={BENCH_FRAMES}, 预热={WARMUP_FRAMES}\n"
    final_content += "=" * 80 + "\n"

    print(f"即将开始 8 个模型的全指标采集...")

    for model in MODELS_TO_TEST:
        m_path = model["path"]
        m_name = model["name"]

        if not os.path.exists(m_path):
            print(f"跳过: 找不到模型 {m_path}")
            continue

        print(f"正在采集: {m_name} ...", end="", flush=True)

        cmd = [
            "python3", BENCHMARK_SCRIPT,
            "--model", m_path,
            "--image", TEST_IMAGE,
            "--bench", str(BENCH_FRAMES),
            "--warmup", str(WARMUP_FRAMES)
        ]

        # 执行并捕获输出
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode == 0:
            model_report = format_model_report(m_name, res.stdout)
            final_content += model_report
            print(" ✅ 已完成")
        else:
            final_content += f"\n### 模型名称: {m_name} (测试运行失败)\n"
            print(" ❌ 运行失败")

    # 保存到文件
    with open(SAVE_RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(final_content)

    print("\n" + "=" * 50)
    print(f"所有模型数据已采集完毕！")
    print(f"结果已保存至: {os.path.abspath(SAVE_RESULT_FILE)}")
    print("=" * 50)


if __name__ == "__main__":
    run_main()