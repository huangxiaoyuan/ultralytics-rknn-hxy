import os
import time
import argparse
import numpy as np
import cv2
import platform
from rknnlite.api import RKNNLite

# ========== 常量定义 ==========
OBJ_THRESH = 0.6
NMS_THRESH = 0.45
IMG_SIZE = (320, 320)

CLASSES = ("accipiter_nisus", "arenaria_interpres", "calidris_falcinellus",
           "calidris_tenuirostris", "calliope_calliope", "centropus_sinensis",
           "circus_spilonotus", "egetta_eulophotes", "egretta_sacra",
           "elanus_caeruleus", "falco_amurensis", "falco_tinnunculus",
           "garrulax_canorus", "halcyon_smyrnensis", "hydrophasianus_chirurgus",
           "leiothrix_argentauris", "leiothrix_lutea", "limnodromus_semipalmatus",
           "merops_philippinus", "milvus_migrans", "numenius_arquata",
           "pandion_haliaetus", "platalea_leucorodia", "platalea_minor")

# ========== 参数解析 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bird_rknn_model/320/best_yolov12_320_op13.rknn")
parser.add_argument("--image", default="platalea_minor.jpg")
parser.add_argument("--bench", default=100, type=int, help="压测帧数")
parser.add_argument("--warmup", default=10, type=int, help="预热帧数")
parser.add_argument("--conf", default=OBJ_THRESH, type=float)
args = parser.parse_args()


# ========== 图像预处理 (Letterbox) ==========
def letterbox(im, new_shape=IMG_SIZE, color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[1] / shape[0], new_shape[0] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[0] - new_unpad[0]) / 2, (new_shape[1] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


# ========== 后处理 (含 Sigmoid) ==========
def post_process(outputs, conf_thresh=OBJ_THRESH):
    pred = outputs[0][0].transpose(1, 0)
    boxes_xywh = pred[:, :4]
    class_logits = pred[:, 4:]

    # 核心激活函数
    class_scores = 1 / (1 + np.exp(-class_logits))

    scores = np.max(class_scores, axis=-1)
    classes = np.argmax(class_scores, axis=-1)

    mask = scores > conf_thresh
    if not np.any(mask): return None, None, None

    # xywh to xyxy
    boxes_xyxy = np.copy(boxes_xywh[mask])
    boxes_xyxy[:, 0] = boxes_xywh[mask, 0] - boxes_xywh[mask, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[mask, 1] - boxes_xywh[mask, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[mask, 0] + boxes_xywh[mask, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[mask, 1] + boxes_xywh[mask, 3] / 2

    # NMS (简化版逻辑)
    final_boxes, final_classes, final_scores = [], [], []
    for c in set(classes[mask]):
        inds = np.where(classes[mask] == c)
        b, s = boxes_xyxy[inds], scores[mask][inds]
        keep = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), conf_thresh, NMS_THRESH)
        if len(keep) > 0:
            for i in keep.flatten():
                final_boxes.append(b[i]);
                final_classes.append(c);
                final_scores.append(s[i])

    if not final_boxes: return None, None, None
    return np.array(final_boxes), np.array(final_classes), np.array(final_scores)


# ========== 核心评估报告函数 ==========
def print_detailed_benchmark(times_infer, times_post, times_total):
    """
    计算并输出极致详细的耗时统计报告
    """
    W = '=' * 72
    print(f'\n{W}')
    print(f'{"RK3588 NPU 推理性能详细报告 ":^72}')
    print(f'{W}')
    print(
        f'{"阶段 (ms)":<15} | {"Min":>7} | {"Max":>7} | {"Mean":>7} | {"P50":>7} | {"P90":>7} | {"P99":>7} | {"Std":>5}')
    print(f'{"-" * 16}|{"-" * 9}|{"-" * 9}|{"-" * 9}|{"-" * 9}|{"-" * 9}|{"-" * 9}|{"-" * 6}')

    def get_stats(arr):
        return {
            "min": np.min(arr), "max": np.max(arr), "mean": np.mean(arr),
            "p50": np.median(arr), "p90": np.percentile(arr, 90),
            "p99": np.percentile(arr, 99), "std": np.std(arr)
        }

    for name, data in [("NPU Inference", times_infer),
                       ("Post-Process", times_post),
                       ("Total Pipeline", times_total)]:
        s = get_stats(data)
        print(f'{name:<15} | {s["min"]:>7.2f} | {s["max"]:>7.2f} | {s["mean"]:>7.2f} | '
              f'{s["p50"]:>7.2f} | {s["p90"]:>7.2f} | {s["p99"]:>7.2f} | {s["std"]:>5.1f}')

    print(f'{W}')

    # FPS 计算
    avg_fps = 1000.0 / np.mean(times_total)
    p90_fps = 1000.0 / np.percentile(times_total, 90)
    stability = (1 - (np.std(times_total) / np.mean(times_total))) * 100

    print(f'  平均吞吐量 (Mean FPS):    {avg_fps:>6.1f} FPS')
    print(f'  延迟保证 (P90 FPS):       {p90_fps:>6.1f} FPS')
    print(f'  运行稳定性 (Stability):   {stability:>6.1f} %  (100% 为绝对平稳)')

    # 抖动分析
    jitter = np.abs(np.diff(times_infer))
    print(f'  帧间抖动 (Max Jitter):    {np.max(jitter):>6.2f} ms')
    print(f'{W}\n')


# ========== 主程序 ==========
if __name__ == '__main__':
    rknn_lite = RKNNLite()

    print(f'--> 加载模型: {args.model}')
    if rknn_lite.load_rknn(args.model) != 0: exit(-1)

    print('--> 初始化 Runtime (Core 0)')
    rknn_lite.init_runtime(target='rk3588', core_mask=RKNNLite.NPU_CORE_0)#core_mask=RKNN.NPU_CORE_0_1_2

    img_src = cv2.imread(args.image)
    img_padded, ratio, (dw, dh) = letterbox(img_src, IMG_SIZE)
    img_input = np.expand_dims(cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB), 0)

    # 1. 预热 (Warmup)
    # 预热非常重要，让 NPU 频率爬升并稳定在高性能模式
    print(f'--> 正在预热 {args.warmup} 帧...')
    for _ in range(args.warmup):
        rknn_lite.inference(inputs=[img_input])

    # 2. 压测 (Benchmark)
    print(f'--> 正在进行压测 {args.bench} 帧...')
    ti_list, tp_list, tt_list = [], [], []

    for i in range(args.bench):
        t0 = time.perf_counter()

        # NPU 推理阶段
        outputs = rknn_lite.inference(inputs=[img_input])
        t1 = time.perf_counter()

        # 后处理阶段
        post_process(outputs, conf_thresh=args.conf)
        t2 = time.perf_counter()

        # 记录耗时 (ms)
        infer_time = (t1 - t0) * 1000
        post_time = (t2 - t1) * 1000
        ti_list.append(infer_time)
        tp_list.append(post_time)
        tt_list.append(infer_time + post_time)

        if (i + 1) % 20 == 0:
            print(f'    已完成 {i + 1}/{args.bench} 帧...')

    # 3. 输出详细报告
    print_detailed_benchmark(np.array(ti_list), np.array(tp_list), np.array(tt_list))

    rknn_lite.release()