"""
1. yolov12 onnx inference
2. built by huangxiaoyuan, 07.19.2025
3. fix bug: input shape, 07.28.2025
4. add: comprehensive inference speed benchmark, 08.xx.2025
5. fix: total = inference + postprocess only, add P50/P90/P99, fix providers bug
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import collections
import statistics

try:
    import spacemit_ort
    _HAS_SPACEMIT = True
except ImportError:
    _HAS_SPACEMIT = False


# ─────────────────────────────────────────────
#  百分位计算
# ─────────────────────────────────────────────
def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


# ─────────────────────────────────────────────
#  性能计时器：记录每个阶段的耗时
# ─────────────────────────────────────────────
class StageTimer:
    """
    轻量级分阶段计时器。
    用法：
        timer = StageTimer()
        timer.start("preprocess")
        ...
        timer.stop("preprocess")
        timer.report()
    """

    STAGE_ORDER = ["preprocess", "inference", "postprocess", "draw"]

    def __init__(self, history_len: int = 100):
        self._t0: dict = {}
        self.history: dict[str, collections.deque] = {
            s: collections.deque(maxlen=history_len) for s in self.STAGE_ORDER
        }
        self.history["total"] = collections.deque(maxlen=history_len)

    # ── 基本操作 ────────────────────────────────
    def start(self, stage: str):
        self._t0[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        elapsed = (time.perf_counter() - self._t0[stage]) * 1000.0
        self.history[stage].append(elapsed)
        return elapsed

    def record_total(self, ms: float):
        self.history["total"].append(ms)

    # ── 统计计算 ────────────────────────────────
    def _stats(self, stage: str) -> dict:
        data = list(self.history[stage])
        if not data:
            return {}
        avg = statistics.mean(data)
        return {
            "n":   len(data),
            "avg": avg,
            "min": min(data),
            "max": max(data),
            "std": statistics.stdev(data) if len(data) > 1 else 0.0,
            "p50": _percentile(data, 50),
            "p90": _percentile(data, 90),
            "p99": _percentile(data, 99),
            "fps": 1000.0 / avg if avg > 0 else 0,
        }

    # ── 打印报告 ────────────────────────────────
    def report(self, title: str = "Performance Report"):
        W    = 79
        sep  = "=" * W
        thin = "-" * W

        # 阶段显示名映射
        labels = {
            "preprocess":  "Pre-Process",
            "inference":   "NPU Inference",
            "postprocess": "Post-Process",
            "draw":        "Draw Results",
            "total":       "Total Pipeline",
        }

        # 取 total FPS
        total_s = self._stats("total")
        fps_str = f"{total_s['fps']:.1f}" if total_s else "N/A"

        print(f"\n{sep}")
        print(f"  {title}")
        print(f"  平均吞吐量 (Total Pipeline): {fps_str} FPS")
        print(f"  注: Total Pipeline = NPU Inference + Post-Process")
        print(sep)

        header = (
            f"  {'阶段 (ms)':<16}| "
            f"{'Min':>7} | {'Max':>7} | {'Mean':>7} | "
            f"{'P50':>7} | {'P90':>7} | {'P99':>7} | {'Std':>5}"
        )
        print(header)
        print(f"  {thin}")

        for stage in self.STAGE_ORDER + ["total"]:
            s = self._stats(stage)
            if not s:
                continue
            label = labels.get(stage, stage)
            row = (
                f"  {label:<16}| "
                f"{s['min']:>7.2f} | {s['max']:>7.2f} | {s['avg']:>7.2f} | "
                f"{s['p50']:>7.2f} | {s['p90']:>7.2f} | {s['p99']:>7.2f} | "
                f"{s['std']:>5.1f}"
            )
            print(row)

        print(f"  {thin}\n")

    def last(self, stage: str) -> float:
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
        benchmark_history: int = 100,
        warmup_runs: int = 0,
    ):
        self.conf_thres  = conf_thres
        self.iou_thres   = iou_thres
        self.score_thres = score_thres
        self.warmup_runs = warmup_runs

        # ── 计时器 ──────────────────────────────
        self.timer = StageTimer(history_len=benchmark_history)

        # ── ONNX Runtime 会话 ───────────────────
        if _HAS_SPACEMIT:
            try:
                self.session = ort.InferenceSession(
                    onnx_path,
                    providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"],
                    provider_options=[{}],
                )
                print("[Info] 使用 SpaceMITExecutionProvider")
            except Exception as e:
                print(f"[Warning] SpaceMIT 失败，回退 CPU: {e}")
                self.session = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"]
                )
        else:
            self.session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            print("[Info] 使用 CPUExecutionProvider")

        # ── 输入信息 ────────────────────────────
        model_inputs            = self.session.get_inputs()
        self.input_name         = model_inputs[0].name
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
    #  预热
    # ──────────────────────────────────────────
    def _warmup(self):
        if self.warmup_runs <= 0:
            return
        print(f"[Warmup] 预热 {self.warmup_runs} 次...")
        dummy    = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        out_name = self.session.get_outputs()[0].name
        for i in range(self.warmup_runs):
            t0 = time.perf_counter()
            self.session.run([out_name], {self.input_name: dummy})
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  warmup [{i+1}/{self.warmup_runs}]: {elapsed:.2f} ms")
            time.sleep(0.05)
        print("[Warmup] 完成\n")

    # ──────────────────────────────────────────
    #  预处理
    # ──────────────────────────────────────────
    def preprocess(self, image: np.ndarray):
        img_h, img_w, _ = image.shape
        scale   = min(self.input_height / img_h, self.input_width / img_w)
        new_w   = int(img_w * scale)
        new_h   = int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_h  = (self.input_height - new_h) // 2
        pad_w  = (self.input_width  - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        input_tensor = (canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, scale, pad_w, pad_h

    # ──────────────────────────────────────────
    #  后处理
    # ──────────────────────────────────────────
    def postprocess(self, output, original_shape, scale, pad_w, pad_h):
        img_h, img_w = original_shape
        predictions  = np.squeeze(output[0]).T       # (N, 4+classes)

        box_preds    = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_scores = 1 / (1 + np.exp(-class_scores))  # sigmoid

        max_class_scores = np.max(class_scores, axis=1)
        class_ids        = np.argmax(class_scores, axis=1)

        keep = max_class_scores > self.conf_thres
        if not np.any(keep):
            return [], [], []

        final_boxes_raw = box_preds[keep]
        final_scores    = max_class_scores[keep]
        final_class_ids = class_ids[keep]

        x1 = final_boxes_raw[:, 0] - final_boxes_raw[:, 2] / 2
        y1 = final_boxes_raw[:, 1] - final_boxes_raw[:, 3] / 2
        x2 = final_boxes_raw[:, 0] + final_boxes_raw[:, 2] / 2
        y2 = final_boxes_raw[:, 1] + final_boxes_raw[:, 3] / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

        cv_boxes  = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes]
        cv_scores = final_scores.tolist()
        indices   = cv2.dnn.NMSBoxes(cv_boxes, cv_scores, self.conf_thres, self.iou_thres)

        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            return boxes[indices].astype(int), final_scores[indices], final_class_ids[indices]
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
            cv2.putText(image, label, (x1, y1 - base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image

    # ──────────────────────────────────────────
    #  主检测入口
    # ──────────────────────────────────────────
    def detect(self, image_path: str, verbose: bool = True) -> np.ndarray | None:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"[Error] 无法读取图片：{image_path}")
            return None

        original_shape = original_image.shape[:2]

        # ── 预处理（不计入 total）──────────────
        self.timer.start("preprocess")
        input_tensor, scale, pad_w, pad_h = self.preprocess(original_image)
        ms_pre = self.timer.stop("preprocess")

        # ── 推理 + 后处理 计入 total ───────────
        t_total_start = time.perf_counter()

        out_name = self.session.get_outputs()[0].name
        self.timer.start("inference")
        outputs  = self.session.run([out_name], {self.input_name: input_tensor})
        ms_inf   = self.timer.stop("inference")

        self.timer.start("postprocess")
        boxes, scores, class_ids = self.postprocess(
            outputs, original_shape, scale, pad_w, pad_h
        )
        ms_post = self.timer.stop("postprocess")

        # total = inference + postprocess
        ms_total = (time.perf_counter() - t_total_start) * 1000
        self.timer.record_total(ms_total)

        # ── 绘图（不计入 total）────────────────
        self.timer.start("draw")
        result_image = self.draw_results(original_image, boxes, scores, class_ids)
        ms_draw = self.timer.stop("draw")

        if verbose:
            print(
                f"[Timing] pre={ms_pre:.1f}ms | "
                f"infer={ms_inf:.1f}ms | "
                f"post={ms_post:.1f}ms | "
                f"draw={ms_draw:.1f}ms | "
                f"total(inf+post)={ms_total:.1f}ms | "
                f"FPS={1000/ms_total:.1f}"
            )

        return result_image

    # ──────────────────────────────────────────
    #  批量压测
    # ──────────────────────────────────────────
    def benchmark(self, image_path: str, runs: int = 50, verbose_per_frame: bool = False):
        print(f"\n[Benchmark] 开始对 '{image_path}' 连续推理 {runs} 次 ...")
        for i in range(runs):
            self.detect(image_path, verbose=verbose_per_frame)
            if not verbose_per_frame and (i + 1) % 10 == 0:
                print(f"  已完成 {i+1}/{runs} 帧")
        self.timer.report(title=f"Benchmark Report ({runs} runs)")


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 ONNX Inference with Benchmark")
    parser.add_argument("--model",  default="/home/yolov12/npu_models/320/best_yolov12n_320.onnx")
    parser.add_argument("--image",  default="Platalea_minor_095.jpg")
    parser.add_argument("--size",   default=320, type=int)
    parser.add_argument("--bench",  default=50,  type=int, help="压测帧数（0=单张推理）")
    parser.add_argument("--warmup", default=0,   type=int)
    args = parser.parse_args()

    yolo_detector = YOLOv12_ONNX_Inference(
        onnx_path         = args.model,
        input_size        = (args.size, args.size),
        warmup_runs       = args.warmup,
        benchmark_history = max(args.bench, 100),
    )

    if args.bench > 0:
        yolo_detector.benchmark(args.image, runs=args.bench, verbose_per_frame=False)
    else:
        result_img = yolo_detector.detect(args.image, verbose=True)
        if result_img is not None:
            cv2.imwrite("detection_result.jpg", result_img)
            cv2.imshow("Detection Result", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()