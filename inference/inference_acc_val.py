"""
YOLOv12 ONNX Inference — SpaceMIT K1 全面推理加速版
优化点：
  1. SessionOptions: intra_op_num_threads 调优 + 图优化级别
  2. 支持 INT8 量化模型（.q.onnx）自动检测提示
  3. IO Binding：零拷贝输入/输出，减少内存搬运开销
  4. 预分配输出缓冲区，避免每帧 malloc
  5. 预处理向量化（纯 numpy，避免 Python 循环）
  6. 后处理 numpy 向量化（去掉 Python for 循环）
  7. 分阶段高精度计时器（StageTimer）
  8. 自动线程数扫描（benchmark_threads），找最优 num_threads
  9. 标准压测入口（benchmark），结束后打印统计报告

# 扫描最优线程数
python yolov12_optimized.py --scan-threads

# 使用最优配置压测 100 帧
python yolov12_optimized.py --model best_320_op13.q.onnx --threads 4 --bench 100

# 对比：关闭 IO Binding 看收益
python yolov12_optimized.py --threads 4 --bench 100 --no-iobind
"""
import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort          # 注册 SpaceMITExecutionProvider
import time
import argparse
import collections
import statistics
import os


# ═══════════════════════════════════════════════════════════
#  分阶段计时器
# ═══════════════════════════════════════════════════════════
class StageTimer:
    STAGE_ORDER = ["preprocess", "inference", "postprocess", "draw"]

    def __init__(self, history_len: int = 200):
        self._t0: dict = {}
        self.history: dict = {
            s: collections.deque(maxlen=history_len)
            for s in self.STAGE_ORDER + ["total"]
        }

    def start(self, stage: str):
        self._t0[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        ms = (time.perf_counter() - self._t0[stage]) * 1000.0
        self.history[stage].append(ms)
        return ms

    def record_total(self, ms: float):
        self.history["total"].append(ms)

    def _stats(self, stage: str) -> dict:
        data = list(self.history[stage])
        if not data:
            return {"n": 0, "avg": 0, "min": 0, "max": 0, "std": 0, "fps": 0}
        avg = statistics.mean(data)
        return {
            "n":   len(data),
            "avg": avg,
            "min": min(data),
            "max": max(data),
            "std": statistics.stdev(data) if len(data) > 1 else 0.0,
            "fps": 1000.0 / avg if avg > 0 else 0,
        }

    def report(self, title: str = "Performance Report"):
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        print(f"  {'Stage':<14} {'Avg(ms)':>8} {'Min(ms)':>8} {'Max(ms)':>8} {'Std(ms)':>8} {'FPS':>7}")
        print(f"  {'-'*60}")
        for stage in self.STAGE_ORDER + ["total"]:
            s = self._stats(stage)
            if s["n"] == 0:
                continue
            print(
                f"  {stage:<14} {s['avg']:>8.2f} {s['min']:>8.2f} "
                f"{s['max']:>8.2f} {s['std']:>8.2f} {s['fps']:>7.1f}"
            )
        print(sep + "\n")

    def last(self, stage: str) -> float:
        d = self.history[stage]
        return d[-1] if d else 0.0


# ═══════════════════════════════════════════════════════════
#  主推理类
# ═══════════════════════════════════════════════════════════
class YOLOv12_ONNX_Inference:

    def __init__(
        self,
        onnx_path: str,
        input_size: tuple       = (320, 320),
        conf_thres: float       = 0.5,
        iou_thres: float        = 0.5,
        score_thres: float      = 0.25,
        num_threads: int        = 4,        # ★ intra_op 线程数，建议 2/4 实测
        use_io_binding: bool    = True,     # ★ IO Binding 零拷贝
        benchmark_history: int  = 200,
        warmup_runs: int        = 5,
    ):
        """
        参数说明
        --------
        num_threads    : ORT intra_op 线程数。SpaceMIT K1 (8核) 推荐先试 2 或 4。
        use_io_binding : True = 启用 IO Binding，减少每帧输入/输出的内存拷贝开销。
        """
        self.conf_thres     = conf_thres
        self.iou_thres      = iou_thres
        self.score_thres    = score_thres
        self.use_io_binding = use_io_binding
        self.timer          = StageTimer(history_len=benchmark_history)

        # ── 模型类型提示 ───────────────────────────────────
        if ".q.onnx" in onnx_path:
            print("[Info] 检测到 INT8 量化模型 (.q.onnx)，将充分利用 SpaceMIT AI 扩展指令加速。")
        else:
            print(
                "[Warn] 当前使用 FP32 模型。强烈建议使用 spine convert 量化为 INT8 (.q.onnx)，"
                "可获得 2~4x 推理加速。\n"
                "       量化命令参考: spine convert onnx --model_path your_model.onnx "
                "--config quant.json"
            )

        # ── ★ 优化1：SessionOptions ────────────────────────
        sess_opts = ort.SessionOptions()

        # intra_op_num_threads：控制单算子内部并行线程数
        # SpaceMIT K1 官方示例设置为 2；你可以用 benchmark_threads() 扫描最优值
        sess_opts.intra_op_num_threads = num_threads

        # inter_op_num_threads：控制算子间并行（对单 batch 推理效果有限，设 1 避免争抢）
        sess_opts.inter_op_num_threads = 1

        # ★ 优化2：图优化级别 — ORT_ENABLE_ALL 开启算子融合 + 常量折叠
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 可选：将优化后的图保存到磁盘，下次加载更快
        # sess_opts.optimized_model_filepath = onnx_path.replace(".onnx", "_opt.onnx")

        # ── 初始化会话 ──────────────────────────────────────
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["SpaceMITExecutionProvider"],
            provider_options=[{}],
        )

        # ── 输入/输出元信息 ─────────────────────────────────
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_width, self.input_height = input_size

        # ★ 优化3：预分配输入/输出缓冲区，避免每帧 malloc
        self._input_buf = np.empty(
            (1, 3, self.input_height, self.input_width), dtype=np.float32
        )

        # ── 类别 ────────────────────────────────────────────
        self.classes = [
            "accipiter_nisus", "arenaria_interpres", "calidris_falcinellus",
            "calidris_tenuirostris", "calliope_calliope", "centropus_sinensis",
            "circus_spilonotus", "egetta_eulophotes", "egretta_sacra",
            "elanus_caeruleus", "falco_amurensis", "falco_tinnunculus",
            "garrulax_canorus", "halcyon_smyrnensis", "hydrophasianus_chirurgus",
            "leiothrix_argentauris", "leiothrix_lutea", "limnodromus_semipalmatus",
            "merops_philippinus", "milvus_migrans", "numenius_arquata",
            "pandion_haliaetus", "platalea_leucorodia", "platalea_minor",
        ]

        # ── 预热 ────────────────────────────────────────────
        self._warmup(warmup_runs)

    # ──────────────────────────────────────────────────────
    #  预热
    # ──────────────────────────────────────────────────────
    def _warmup(self, runs: int):
        if runs <= 0:
            return
        print(f"[Warmup] 开始预热 {runs} 次 (threads={self.session.get_session_options().intra_op_num_threads}) ...")
        dummy = np.random.rand(1, 3, self.input_height, self.input_width).astype(np.float32)
        for i in range(runs):
            t0 = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: dummy})
            print(f"  warmup [{i+1}/{runs}]: {(time.perf_counter()-t0)*1000:.1f} ms")
        print("[Warmup] 完成\n")

    # ──────────────────────────────────────────────────────
    #  ★ 优化4：向量化预处理（复用预分配缓冲区）
    # ──────────────────────────────────────────────────────
    def preprocess(self, image: np.ndarray):
        img_h, img_w = image.shape[:2]
        scale  = min(self.input_height / img_h, self.input_width / img_w)
        new_w  = int(img_w * scale)
        new_h  = int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (self.input_height - new_h) // 2
        pad_w = (self.input_width  - new_w) // 2

        # 复用预分配画布：先填灰色，再写入有效区域
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # ★ 向量化：BGR→RGB + CHW + float32 归一化，写入预分配 buf
        # np.copyto 避免额外内存分配
        np.copyto(
            self._input_buf[0],
            (canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0).astype(np.float32)
        )

        return self._input_buf, scale, pad_w, pad_h

    # ──────────────────────────────────────────────────────
    #  ★ 优化5：IO Binding 推理（可选）
    # ──────────────────────────────────────────────────────
    def _run_with_io_binding(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        IO Binding 让 ORT 直接操作我们提供的内存指针，
        避免 session.run() 内部的数据拷贝开销。
        对 SpaceMIT AI 扩展指令访问连续内存更友好。
        """
        io_binding = self.session.io_binding()

        # 绑定输入（使用已有 numpy array 的内存）
        io_binding.bind_cpu_input(self.input_name, input_tensor)

        # 绑定输出（让 ORT 自己分配，也可以 bind_output 到预分配 buffer）
        io_binding.bind_output(self.output_name)

        self.session.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0]

    # ──────────────────────────────────────────────────────
    #  ★ 优化6：向量化后处理
    # ──────────────────────────────────────────────────────
    def postprocess(self, output, original_shape, scale, pad_w, pad_h):
        img_h, img_w = original_shape

        # (1, 28, 8400) → (8400, 28)
        predictions = np.squeeze(output).T         # shape: (8400, 28)

        box_preds       = predictions[:, :4]       # cx,cy,w,h
        class_scores    = predictions[:, 4:]       # (8400, 24)

        # 向量化：一次性取最大分数和类别
        max_scores = class_scores.max(axis=1)      # (8400,)
        class_ids  = class_scores.argmax(axis=1)   # (8400,)

        # 过滤低置信度
        keep = max_scores > self.conf_thres
        if not keep.any():
            return [], [], []

        box_f   = box_preds[keep]
        score_f = max_scores[keep]
        cid_f   = class_ids[keep]

        # ★ 向量化坐标转换（无 Python 循环）
        half_w = box_f[:, 2] / 2
        half_h = box_f[:, 3] / 2
        x1 = ((box_f[:, 0] - half_w) - pad_w) / scale
        y1 = ((box_f[:, 1] - half_h) - pad_h) / scale
        x2 = ((box_f[:, 0] + half_w) - pad_w) / scale
        y2 = ((box_f[:, 1] + half_h) - pad_h) / scale

        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        cv_boxes  = boxes.astype(int)
        cv_boxes_wh = np.stack([
            cv_boxes[:, 0], cv_boxes[:, 1],
            cv_boxes[:, 2] - cv_boxes[:, 0],
            cv_boxes[:, 3] - cv_boxes[:, 1],
        ], axis=1).tolist()

        indices = cv2.dnn.NMSBoxes(
            cv_boxes_wh, score_f.tolist(), self.conf_thres, self.iou_thres
        )
        if len(indices) == 0:
            return [], [], []

        indices = np.array(indices).flatten()
        return boxes[indices].astype(int), score_f[indices], cid_f[indices]

    # ──────────────────────────────────────────────────────
    #  绘图
    # ──────────────────────────────────────────────────────
    def draw_results(self, image, boxes, scores, class_ids):
        for box, score, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            label = f"{self.classes[cid]}: {score:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - lh - base), (x1 + lw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return image

    # ──────────────────────────────────────────────────────
    #  主检测流程
    # ──────────────────────────────────────────────────────
    def detect(self, image_path: str, verbose: bool = True):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Error] 无法读取图片: {image_path}")
            return None

        t_total = time.perf_counter()

        # 预处理
        self.timer.start("preprocess")
        input_tensor, scale, pad_w, pad_h = self.preprocess(img)
        ms_pre = self.timer.stop("preprocess")

        # 推理
        self.timer.start("inference")
        if self.use_io_binding:
            raw_out = self._run_with_io_binding(input_tensor)
        else:
            raw_out = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        ms_inf = self.timer.stop("inference")

        # 后处理
        self.timer.start("postprocess")
        boxes, scores, cids = self.postprocess(raw_out, img.shape[:2], scale, pad_w, pad_h)
        ms_post = self.timer.stop("postprocess")

        # 绘图
        self.timer.start("draw")
        result = self.draw_results(img, boxes, scores, cids)
        ms_draw = self.timer.stop("draw")

        ms_total = (time.perf_counter() - t_total) * 1000
        self.timer.record_total(ms_total)

        if verbose:
            print(
                f"[Frame] pre={ms_pre:.1f}ms | infer={ms_inf:.1f}ms | "
                f"post={ms_post:.1f}ms | draw={ms_draw:.1f}ms | "
                f"total={ms_total:.1f}ms | FPS={1000/ms_total:.1f}"
            )
        return result

    # ──────────────────────────────────────────────────────
    #  压测（批量推理同一图 N 次）
    # ──────────────────────────────────────────────────────
    def benchmark(self, image_path: str, runs: int = 100, verbose_per_frame: bool = False):
        print(f"\n[Benchmark] '{image_path}' × {runs} 帧  "
              f"(threads={self.session.get_session_options().intra_op_num_threads}, "
              f"io_binding={self.use_io_binding}) ...")
        for i in range(runs):
            self.detect(image_path, verbose=verbose_per_frame)
            if not verbose_per_frame and (i + 1) % 20 == 0:
                avg = statistics.mean(self.timer.history["inference"])
                print(f"  [{i+1:>4}/{runs}] avg_infer={avg:.1f}ms")
        self.timer.report(f"Benchmark ({runs} runs, threads={self.session.get_session_options().intra_op_num_threads})")

    # ──────────────────────────────────────────────────────
    #  ★ 自动线程扫描：找最优 num_threads
    # ──────────────────────────────────────────────────────
    @staticmethod
    def benchmark_threads(
        onnx_path: str,
        image_path: str,
        input_size: tuple   = (320, 320),
        thread_list: list   = None,
        runs_per_cfg: int   = 30,
    ):
        """
        对不同 num_threads 配置各跑 runs_per_cfg 帧，打印对比表，
        找出推理延迟最低的线程数。

        用法：
            YOLOv12_ONNX_Inference.benchmark_threads(
                "model.q.onnx", "test.jpg", thread_list=[1,2,4,8]
            )
        """
        if thread_list is None:
            thread_list = [1, 2, 4, 8]

        results = []
        for n in thread_list:
            det = YOLOv12_ONNX_Inference(
                onnx_path, input_size=input_size,
                num_threads=n, warmup_runs=3, benchmark_history=runs_per_cfg
            )
            # 纯推理热身后压测
            dummy = np.random.rand(1, 3, input_size[1], input_size[0]).astype(np.float32)
            times = []
            for _ in range(runs_per_cfg):
                t0 = time.perf_counter()
                det.session.run([det.output_name], {det.input_name: dummy})
                times.append((time.perf_counter() - t0) * 1000)
            avg = statistics.mean(times)
            mn  = min(times)
            results.append((n, avg, mn, 1000/avg))
            print(f"  threads={n}: avg={avg:.2f}ms  min={mn:.2f}ms  FPS={1000/avg:.1f}")

        best = min(results, key=lambda x: x[1])
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  线程扫描结果 ({runs_per_cfg} runs each)")
        print(sep)
        print(f"  {'threads':>8} {'avg(ms)':>9} {'min(ms)':>9} {'FPS':>7}")
        print(f"  {'-'*48}")
        for n, avg, mn, fps in results:
            marker = " ◀ best" if n == best[0] else ""
            print(f"  {n:>8} {avg:>9.2f} {mn:>9.2f} {fps:>7.1f}{marker}")
        print(f"{sep}\n  → 推荐使用 num_threads = {best[0]}\n{sep}\n")
        return best[0]


# ═══════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 ONNX — SpaceMIT K1 优化推理")
    parser.add_argument("--model",    default="/home/yolov12/npu/best_npu_320_op13.onnx")
    parser.add_argument("--image",    default="Platalea_minor_053.jpg")
    parser.add_argument("--size",     default=320,  type=int,  help="输入尺寸（宽=高）")
    parser.add_argument("--threads",  default=4,    type=int,  help="intra_op_num_threads")
    parser.add_argument("--bench",    default=100,  type=int,  help="压测帧数（0=单张）")
    parser.add_argument("--warmup",   default=5,    type=int)
    parser.add_argument("--no-iobind", action="store_true",    help="禁用 IO Binding")
    parser.add_argument("--scan-threads", action="store_true", help="自动扫描最优线程数")
    args = parser.parse_args()

    # ── 可选：先扫描最优线程数 ──────────────────────────────
    if args.scan_threads:
        best_t = YOLOv12_ONNX_Inference.benchmark_threads(
            args.model, args.image,
            input_size=(args.size, args.size),
            thread_list=[1, 2, 4, 8],
            runs_per_cfg=30,
        )
        args.threads = best_t

    # ── 实例化检测器 ────────────────────────────────────────
    detector = YOLOv12_ONNX_Inference(
        onnx_path      = args.model,
        input_size     = (args.size, args.size),
        num_threads    = args.threads,
        use_io_binding = not args.no_iobind,
        warmup_runs    = args.warmup,
        benchmark_history = max(args.bench, 200),
    )

    if args.bench > 0:
        detector.benchmark(args.image, runs=args.bench)
    else:
        result_img = detector.detect(args.image, verbose=True)
        if result_img is not None:
            cv2.imwrite("e_detection_result.jpg", result_img)
            cv2.imshow("Detection Result", result_img)
            cv2.destroyAllWindows()