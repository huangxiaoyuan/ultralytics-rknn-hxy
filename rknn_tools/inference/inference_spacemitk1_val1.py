"""
1. yolov12 onnx inference
2. built by huangxiaoyuan, 07.19.2025
3. fix bug: input shape, 07.28.2025
4. add: comprehensive inference speed benchmark, 08.xx.2025.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort


# ─────────────────────────────────────────────
#  性能计时器：记录每个阶段的耗时
# ─────────────────────────────────────────────
class StageTimer:
    """轻量级分阶段计时器。 用法： timer = StageTimer() timer.start("preprocess") ... timer.stop("preprocess") timer.report().
    """

    STAGE_ORDER = ["preprocess", "inference", "postprocess", "draw"]

    def __init__(self, history_len: int = 100):
        self._t0: dict = {}
        # 滑动窗口，保存最近 history_len 帧的耗时（毫秒）
        self.history: dict[str, collections.deque] = {
            s: collections.deque(maxlen=history_len) for s in self.STAGE_ORDER
        }
        self.history["total"] = collections.deque(maxlen=history_len)

    # ── 基本操作 ────────────────────────────────
    def start(self, stage: str):
        self._t0[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """停止计时并返回本次耗时（ms）."""
        elapsed = (time.perf_counter() - self._t0[stage]) * 1000.0
        self.history[stage].append(elapsed)
        return elapsed

    def record_total(self, ms: float):
        self.history["total"].append(ms)

    # ── 统计计算 ────────────────────────────────
    def _stats(self, stage: str) -> dict:
        data = list(self.history[stage])
        if not data:
            return {"n": 0, "avg": 0, "min": 0, "max": 0, "std": 0, "fps": 0}
        avg = statistics.mean(data)
        return {
            "n": len(data),
            "avg": avg,
            "min": min(data),
            "max": max(data),
            "std": statistics.stdev(data) if len(data) > 1 else 0.0,
            "fps": 1000.0 / avg if avg > 0 else 0,
        }

    # ── 打印报告 ────────────────────────────────
    def report(self, title: str = "Performance Report"):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        print(f"  {'Stage':<14} {'Avg(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8} {'Std(ms)':>8} {'FPS':>7}")
        print(f"  {'-' * 55}")
        for stage in [*self.STAGE_ORDER, "total"]:
            s = self._stats(stage)
            if s["n"] == 0:
                continue
            print(f"  {stage:<14} {s['avg']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f} {s['std']:>8.2f} {s['fps']:>7.1f}")
        print(sep + "\n")

    def last(self, stage: str) -> float:
        """返回某阶段最近一帧的耗时（ms），没有记录则返回 0."""
        d = self.history[stage]
        return d[-1] if d else 0.0


# ─────────────────────────────────────────────
#  主推理类
# ─────────────────────────────────────────────
class YOLOv12_ONNX_Inference:
    def __init__(
        self,
        onnx_path: str,
        input_size: tuple = (320, 320),
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        score_thres: float = 0.25,
        benchmark_history: int = 100,  # 滑动窗口帧数
        warmup_runs: int = 3,  # 预热推理次数
    ):
        """
        初始化 YOLOv12 ONNX 推理器。.

        :param onnx_path:          ONNX 模型文件路径
        :param input_size:         模型输入尺寸 (宽, 高)
        :param conf_thres:         置信度阈值
        :param iou_thres:          NMS IOU 阈值
        :param score_thres:        类别分数阈值
        :param benchmark_history:  滑动窗口保留的最近帧数
        :param warmup_runs:        初始化后自动预热的推理次数（避免首帧异常偏高）
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.score_thres = score_thres
        self.warmup_runs = warmup_runs

        # ── 计时器 ──────────────────────────────
        self.timer = StageTimer(history_len=benchmark_history)

        # ── ONNX Runtime 会话 ───────────────────
        ep_provider_options = {}
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["SpaceMITExecutionProvider"],
            provider_options=[ep_provider_options],
        )

        # ── 输入信息 ────────────────────────────
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_width, self.input_height = input_size

        # ── 类别列表 ────────────────────────────
        self.classes = [
            "accipiter_nisus",
            "arenaria_interpres",
            "calidris_falcinellus",
            "calidris_tenuirostris",
            "calliope_calliope",
            "centropus_sinensis",
            "circus_spilonotus",
            "egetta_eulophotes",
            "egretta_sacra",
            "elanus_caeruleus",
            "falco_amurensis",
            "falco_tinnunculus",
            "garrulax_canorus",
            "halcyon_smyrnensis",
            "hydrophasianus_chirurgus",
            "leiothrix_argentauris",
            "leiothrix_lutea",
            "limnodromus_semipalmatus",
            "merops_philippinus",
            "milvus_migrans",
            "numenius_arquata",
            "pandion_haliaetus",
            "platalea_leucorodia",
            "platalea_minor",
        ]

        # ── 预热 ────────────────────────────────
        self._warmup()

    # ──────────────────────────────────────────
    #  预热：用随机数据跑几次，避免首帧 JIT/初始化延迟
    # ──────────────────────────────────────────
    def _warmup(self):
        if self.warmup_runs <= 0:
            return
        print(f"[Warmup] 正在用随机数据预热 {self.warmup_runs} 次...")
        dummy = np.random.rand(1, 3, self.input_height, self.input_width).astype(np.float32)
        out_name = self.session.get_outputs()[0].name
        for i in range(self.warmup_runs):
            t0 = time.perf_counter()
            self.session.run([out_name], {self.input_name: dummy})
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  warmup [{i + 1}/{self.warmup_runs}]: {elapsed:.2f} ms")
        print("[Warmup] 完成\n")

    # ──────────────────────────────────────────
    #  预处理
    # ──────────────────────────────────────────
    def preprocess(self, image: np.ndarray):
        """Letterbox + HWC→CHW + normalize."""
        img_h, img_w, _ = image.shape

        scale = min(self.input_height / img_h, self.input_width / img_w)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_h = (self.input_height - new_h) // 2
        pad_w = (self.input_width - new_w) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        input_tensor = (canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor, scale, pad_w, pad_h

    # ──────────────────────────────────────────
    #  后处理
    # ──────────────────────────────────────────
    def postprocess(self, output, original_shape, scale, pad_w, pad_h):
        """适配 (batch, attributes, predictions) → boxes/scores/class_ids."""
        img_h, img_w = original_shape

        # (1, 28, 8400) → (8400, 28)
        predictions = np.squeeze(output[0]).T

        len(self.classes)
        box_preds = predictions[:, :4]
        class_scores = predictions[:, 4:]  # (8400, 24)

        max_class_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        keep = max_class_scores > self.conf_thres
        if not np.any(keep):
            return [], [], []

        final_boxes_raw = box_preds[keep]
        final_scores = max_class_scores[keep]
        final_class_ids = class_ids[keep]

        # (cx, cy, w, h) → (x1, y1, x2, y2)
        x1 = final_boxes_raw[:, 0] - final_boxes_raw[:, 2] / 2
        y1 = final_boxes_raw[:, 1] - final_boxes_raw[:, 3] / 2
        x2 = final_boxes_raw[:, 0] + final_boxes_raw[:, 2] / 2
        y2 = final_boxes_raw[:, 1] + final_boxes_raw[:, 3] / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 映射回原始图像坐标
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

        # NMS
        cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        cv_scores = final_scores.tolist()
        indices = cv2.dnn.NMSBoxes(cv_boxes, cv_scores, self.conf_thres, self.iou_thres)

        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            return (
                boxes[indices].astype(int),
                final_scores[indices],
                final_class_ids[indices],
            )
        return [], [], []

    # ──────────────────────────────────────────
    #  绘制结果
    # ──────────────────────────────────────────
    def draw_results(self, image: np.ndarray, boxes, scores, class_ids) -> np.ndarray:
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            label = f"{self.classes[class_id]}: {score:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - lh - base), (x1 + lw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image

    # ──────────────────────────────────────────
    #  主检测入口（含完整分阶段计时）
    # ──────────────────────────────────────────
    def detect(self, image_path: str, verbose: bool = True) -> np.ndarray | None:
        """
        执行完整检测并记录每阶段耗时。.

        :param image_path: 图片路径
        :param verbose:    是否打印本帧各阶段耗时
        :return:           绘制结果的图像（BGR numpy array）
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"[Error] 无法读取图片：{image_path}")
            return None

        original_shape = original_image.shape[:2]
        t_total_start = time.perf_counter()

        # ── 预处理 ──────────────────────────────
        self.timer.start("preprocess")
        input_tensor, scale, pad_w, pad_h = self.preprocess(original_image)
        ms_pre = self.timer.stop("preprocess")

        # ── 推理 ────────────────────────────────
        out_name = self.session.get_outputs()[0].name
        self.timer.start("inference")
        outputs = self.session.run([out_name], {self.input_name: input_tensor})
        ms_inf = self.timer.stop("inference")

        # ── 后处理 ──────────────────────────────
        self.timer.start("postprocess")
        boxes, scores, class_ids = self.postprocess(outputs, original_shape, scale, pad_w, pad_h)
        ms_post = self.timer.stop("postprocess")

        # ── 绘图 ────────────────────────────────
        self.timer.start("draw")
        result_image = self.draw_results(original_image, boxes, scores, class_ids)
        ms_draw = self.timer.stop("draw")

        # ── 总计 ────────────────────────────────
        ms_total = (time.perf_counter() - t_total_start) * 1000
        self.timer.record_total(ms_total)

        if verbose:
            print(
                f"[Timing] pre={ms_pre:.1f}ms | "
                f"infer={ms_inf:.1f}ms | "
                f"post={ms_post:.1f}ms | "
                f"draw={ms_draw:.1f}ms | "
                f"total={ms_total:.1f}ms | "
                f"FPS={1000 / ms_total:.1f}"
            )

        return result_image

    # ──────────────────────────────────────────
    #  批量压测（连续推理同一张图 N 次）
    # ──────────────────────────────────────────
    def benchmark(self, image_path: str, runs: int = 50, verbose_per_frame: bool = False):
        """
        对单张图像连续推理 `runs` 次，结束后打印统计报告。.

        :param image_path:        图片路径
        :param runs:              推理次数
        :param verbose_per_frame: 是否打印每帧耗时
        """
        print(f"\n[Benchmark] 开始对 '{image_path}' 连续推理 {runs} 次 ...")
        for i in range(runs):
            self.detect(image_path, verbose=verbose_per_frame)
            if not verbose_per_frame and (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{runs} 帧")
        self.timer.report(title=f"Benchmark Report ({runs} runs)")


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 ONNX Inference with Benchmark")
    parser.add_argument("--model", default="/home/yolov12/npu/best_320_op13.onnx", help="ONNX 模型路径")
    parser.add_argument("--image", default="Platalea_minor_095.jpg", help="输入图片路径")
    parser.add_argument("--size", default=320, type=int, help="模型输入尺寸（宽=高）")
    parser.add_argument("--bench", default=50, type=int, help="压测帧数（0 表示只跑单张）")
    parser.add_argument("--warmup", default=3, type=int, help="预热推理次数")
    args = parser.parse_args()

    # ── 实例化检测器 ─────────────────────────────
    yolo_detector = YOLOv12_ONNX_Inference(
        onnx_path=args.model,
        input_size=(args.size, args.size),
        warmup_runs=args.warmup,
        benchmark_history=max(args.bench, 100),
    )

    if args.bench > 0:
        # ── 压测模式 ─────────────────────────────
        yolo_detector.benchmark(args.image, runs=args.bench, verbose_per_frame=False)
    else:
        # ── 单张推理模式 ─────────────────────────
        result_img = yolo_detector.detect(args.image, verbose=True)
        if result_img is not None:
            cv2.imwrite("e_detection_result.jpg", result_img)
            cv2.imshow("Detection Result", result_img)
            cv2.destroyAllWindows()
