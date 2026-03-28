import os
import time
import argparse
import numpy as np
import cv2
import platform
from rknnlite.api import RKNNLite

# ========== 常量定义 ==========
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE   = (320, 320)  # (width, height)

CLASSES = ("accipiter_nisus", "arenaria_interpres", "calidris_falcinellus",
           "calidris_tenuirostris", "calliope_calliope", "centropus_sinensis",
           "circus_spilonotus", "egetta_eulophotes", "egretta_sacra",
           "elanus_caeruleus", "falco_amurensis", "falco_tinnunculus",
           "garrulax_canorus", "halcyon_smyrnensis", "hydrophasianus_chirurgus",
           "leiothrix_argentauris", "leiothrix_lutea", "limnodromus_semipalmatus",
           "merops_philippinus", "milvus_migrans", "numenius_arquata",
           "pandion_haliaetus", "platalea_leucorodia", "platalea_minor")

DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'


# ========== 参数解析 ==========
parser = argparse.ArgumentParser(description="YOLOv12 RKNN Inference with Evaluation")
parser.add_argument("--model",    default="bird_rknn_model/yolov12-bird-320_rk3588.rknn")
parser.add_argument("--image",    default="platalea_minor.jpg")
parser.add_argument("--img_save", action="store_true", default=True)
parser.add_argument("--img_show", action="store_true", default=False)
parser.add_argument("--bench",    default=50, type=int)
parser.add_argument("--warmup",   default=3,  type=int)
parser.add_argument("--conf",     default=OBJ_THRESH, type=float,
                    help="置信度阈值（覆盖默认 OBJ_THRESH）")
args = parser.parse_args()


# ========== 平台检测 ==========
def get_host():
    system  = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                s = f.read()
                if 'rk3588' in s:   return 'RK3588'
                if 'rk3562' in s:   return 'RK3562'
                return 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    return os_machine


# ========== 图像预处理 ==========
def letterbox(im, new_shape=IMG_SIZE, color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[1] / shape[0], new_shape[0] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[0] - new_unpad[0]
    dh = new_shape[1] - new_unpad[1]
    dw /= 2;  dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top,  bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(img_src, input_size=IMG_SIZE):
    img, ratio, (dw, dh) = letterbox(img_src, new_shape=input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img, ratio, (dw, dh)


# ========== 后处理 ==========
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes, scores):
    x = boxes[:, 0];  y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0];  h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0];  keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i]+w[i], x[order[1:]]+w[order[1:]])
        yy2 = np.minimum(y[i]+h[i], y[order[1:]]+h[order[1:]])
        w1  = np.maximum(0.0, xx2-xx1+0.00001)
        h1  = np.maximum(0.0, yy2-yy1+0.00001)
        inter = w1 * h1
        ovr   = inter / (areas[i] + areas[order[1:]] - inter)
        inds  = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def post_process(outputs, conf_thresh=OBJ_THRESH):
    pred = outputs[0][0].transpose(1, 0)   # (2100, 28)
    boxes_xywh   = pred[:, :4]
    class_scores = pred[:, 4:]             # 原始 logits，不做 sigmoid
    class_max_score = np.max(class_scores,   axis=-1)
    classes         = np.argmax(class_scores, axis=-1)
    mask            = class_max_score > conf_thresh   # 严格大于，过滤 sigmoid(0)噪声
    boxes_xywh      = boxes_xywh[mask]
    class_max_score = class_max_score[mask]
    classes         = classes[mask]
    if len(boxes_xywh) == 0:
        return None, None, None
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes_xyxy[inds];  s = class_max_score[inds]
        keep = nms_boxes(b, s)
        if len(keep):
            nboxes.append(b[keep]);  nclasses.append(classes[inds][keep]);  nscores.append(s[keep])
    if not nboxes:
        return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def get_real_box(boxes, ratio, dw, dh):
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    b[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    return b


# ========== 可视化 ==========
def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(b) for b in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ========== 评估信息打印 ==========
def print_detection_eval(boxes, scores, classes, img_shape, ratio, dw, dh):
    """打印单张图片的详细检测评估信息"""
    W = '=' * 54

    print(f'\n{W}')
    print(f'  检测评估报告')
    print(f'{W}')

    if boxes is None or len(boxes) == 0:
        print('  未检测到任何目标')
        print(W)
        return

    total = len(boxes)
    print(f'  检测目标总数:  {total}')
    print(f'  置信度阈值:    {args.conf:.2f}')
    print(f'  NMS IoU 阈值:  {NMS_THRESH:.2f}')

    # ── 各类别统计 ──────────────────────────────────────────────
    print(f'\n  {"[ 各类别检测数量 ]":-^48}')
    from collections import Counter
    class_counter = Counter(classes.tolist())
    print(f'  {"类别名称":<32} {"数量":>6}  {"占比":>6}')
    print(f'  {"-"*46}')
    for cl_id, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = '█' * int(pct / 5)
        print(f'  {CLASSES[cl_id]:<32} {count:>6}  {pct:>5.1f}%  {bar}')

    # ── 置信度分布 ───────────────────────────────────────────────
    print(f'\n  {"[ 置信度分布 ]":-^48}')
    print(f'  最高分:   {scores.max():.4f}')
    print(f'  最低分:   {scores.min():.4f}')
    print(f'  平均分:   {scores.mean():.4f}')
    print(f'  中位数:   {np.median(scores):.4f}')
    print(f'  标准差:   {scores.std():.4f}')

    bins = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ['0.25-0.4', '0.4-0.5', '0.5-0.6',
              '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    print(f'\n  {"区间":<12} {"数量":>6}  {"占比":>6}')
    print(f'  {"-"*30}')
    for i, label in enumerate(labels):
        cnt = np.sum((scores >= bins[i]) & (scores < bins[i+1]))
        if cnt > 0:
            pct = cnt / total * 100
            bar = '█' * int(pct / 5)
            print(f'  {label:<12} {cnt:>6}  {pct:>5.1f}%  {bar}')

    # ── 检测框尺寸分布 ───────────────────────────────────────────
    boxes_real = get_real_box(boxes, ratio, dw, dh)
    img_h, img_w = img_shape[:2]
    img_area = img_w * img_h

    bw = boxes_real[:, 2] - boxes_real[:, 0]
    bh = boxes_real[:, 3] - boxes_real[:, 1]
    areas = bw * bh
    rel_areas = areas / img_area * 100   # 占图像面积百分比

    print(f'\n  {"[ 检测框尺寸（原图坐标）]":-^48}')
    print(f'  {"":12} {"宽":>8} {"高":>8} {"面积%":>8}')
    print(f'  {"-"*40}')
    print(f'  {"最小":12} {bw.min():>8.1f} {bh.min():>8.1f} {rel_areas.min():>7.2f}%')
    print(f'  {"最大":12} {bw.max():>8.1f} {bh.max():>8.1f} {rel_areas.max():>7.2f}%')
    print(f'  {"平均":12} {bw.mean():>8.1f} {bh.mean():>8.1f} {rel_areas.mean():>7.2f}%')

    # 按尺寸分类（小/中/大目标，COCO标准）
    small  = np.sum(areas < 32**2)
    medium = np.sum((areas >= 32**2) & (areas < 96**2))
    large  = np.sum(areas >= 96**2)
    print(f'\n  目标尺寸分类 (COCO 标准):')
    print(f'    小目标 (<32²px):    {small:>4} 个  ({small/total*100:.1f}%)')
    print(f'    中目标 (32²~96²px): {medium:>4} 个  ({medium/total*100:.1f}%)')
    print(f'    大目标 (>96²px):    {large:>4} 个  ({large/total*100:.1f}%)')

    # ── 逐框详情 ─────────────────────────────────────────────────
    print(f'\n  {"[ 逐框检测结果 ]":-^48}')
    print(f'  {"#":>3}  {"类别":<28} {"置信度":>7}  {"x1":>5} {"y1":>5} {"x2":>5} {"y2":>5}')
    print(f'  {"-"*68}')
    for i, (box, score, cl) in enumerate(zip(boxes_real, scores, classes)):
        x1, y1, x2, y2 = [int(b) for b in box]
        print(f'  {i+1:>3}  {CLASSES[cl]:<28} {score:>7.4f}  {x1:>5} {y1:>5} {x2:>5} {y2:>5}')

    print(W)


def print_timing_eval(t_preprocess, t_infer, t_post, t_draw,
                      times_infer=None, times_post=None, times_pipeline=None):
    """打印时间评估信息"""
    W = '=' * 54
    t_pipeline = t_preprocess + t_infer + t_post

    print(f'\n{W}')
    print(f'  推理时间评估报告')
    print(f'{W}')
    print(f'  {"阶段":<18} {"耗时":>10}')
    print(f'  {"-"*30}')
    print(f'  {"图像预处理":<18} {t_preprocess:>9.2f} ms')
    print(f'  {"NPU 推理":<18} {t_infer:>9.2f} ms')
    print(f'  {"后处理":<18} {t_post:>9.2f} ms')
    print(f'  {"画框+保存":<18} {t_draw:>9.2f} ms')
    print(f'  {"-"*30}')
    print(f'  {"单帧流水线":<18} {t_pipeline:>9.2f} ms  →  {1000/t_pipeline:.1f} FPS')

    if times_infer is not None:
        print(f'\n  {"[ 压测统计 (N={len(times_infer)}) ]":-^48}')
        print(f'  {"阶段":<14} {"Min":>7} {"Max":>7} {"Mean":>7} {"Median":>8} {"Std":>6}  {"P90":>7}  {"P99":>7}')
        print(f'  {"-"*68}')
        for name, arr in [("NPU推理(ms)",   times_infer),
                          ("后处理(ms)",    times_post),
                          ("流水线(ms)",    times_pipeline)]:
            p90 = np.percentile(arr, 90)
            p99 = np.percentile(arr, 99)
            print(f'  {name:<14} {arr.min():>7.2f} {arr.max():>7.2f} {arr.mean():>7.2f}'
                  f' {np.median(arr):>8.2f} {arr.std():>6.2f}  {p90:>7.2f}  {p99:>7.2f}')
        print(f'  {"-"*68}')
        fps_mean   = 1000 / times_pipeline.mean()
        fps_p90    = 1000 / np.percentile(times_pipeline, 90)
        fps_stable = 1000 / (times_pipeline.mean() + times_pipeline.std())
        print(f'  平均 FPS:          {fps_mean:.1f}')
        print(f'  P90 FPS:           {fps_p90:.1f}')
        print(f'  稳定 FPS (μ+σ):    {fps_stable:.1f}')

        # 帧时间抖动分析
        jitter = np.diff(times_infer)
        print(f'\n  帧间抖动 (Jitter):')
        print(f'    平均抖动:  {np.abs(jitter).mean():.2f} ms')
        print(f'    最大抖动:  {np.abs(jitter).max():.2f} ms')

    print(W)


# ========== 主流程 ==========
if __name__ == '__main__':

    t_total_start = time.perf_counter()

    host_name = get_host()
    if host_name not in ['RK3588', 'RK3562', 'RK3566_RK3568']:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    # 加载模型
    t0 = time.perf_counter()
    rknn_lite = RKNNLite()
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(args.model)
    if ret != 0:
        print('Load RKNN model failed'); exit(ret)
    t_load = (time.perf_counter() - t0) * 1000
    print(f'done  [{t_load:.2f} ms]')

    # 初始化运行时
    t0 = time.perf_counter()
    print('--> Init runtime environment')
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!'); exit(ret)
    t_init = (time.perf_counter() - t0) * 1000
    print(f'done  [{t_init:.2f} ms]')

    # 读取图片
    img_src = cv2.imread(args.image)
    if img_src is None:
        print(f'Image {args.image} not found'); exit(-1)
    print(f'\n图片信息: {args.image}  {img_src.shape[1]}x{img_src.shape[0]}')

    # 预处理
    t0 = time.perf_counter()
    img_input, ratio, (dw, dh) = preprocess(img_src, input_size=IMG_SIZE)
    t_preprocess = (time.perf_counter() - t0) * 1000

    # 预热
    print(f'\n--> Warmup ({args.warmup} times)')
    for _ in range(args.warmup):
        rknn_lite.inference(inputs=[img_input])
    print('done')

    # 单张推理
    print('\n--> Running model')
    t0 = time.perf_counter()
    outputs = rknn_lite.inference(inputs=[img_input])
    t_infer = (time.perf_counter() - t0) * 1000
    print(f'done  [{t_infer:.2f} ms]')

    # 后处理
    t0 = time.perf_counter()
    boxes, classes, scores = post_process(outputs, conf_thresh=args.conf)
    t_post = (time.perf_counter() - t0) * 1000

    # 画框保存
    t0 = time.perf_counter()
    img_p = img_src.copy()
    if boxes is not None:
        boxes_real = get_real_box(boxes, ratio, dw, dh)
        draw(img_p, boxes_real, scores, classes)
    if args.img_save:
        os.makedirs('./result', exist_ok=True)
        save_path = './result/' + os.path.basename(args.image)
        cv2.imwrite(save_path, img_p)
        print(f'Result saved to {save_path}')
    if args.img_show:
        cv2.imshow("result", img_p); cv2.waitKey(0); cv2.destroyAllWindows()
    t_draw = (time.perf_counter() - t0) * 1000

    # ── 评估输出 ──────────────────────────────────────────────────
    print_detection_eval(boxes, scores, classes, img_src.shape, ratio, dw, dh)

    # 压测
    times_infer = times_post = times_pipeline = None
    if args.bench > 0:
        print(f'\n--> Benchmark ({args.bench} frames)')
        ti, tp, tpl = [], [], []
        for _ in range(args.bench):
            t0 = time.perf_counter()
            out = rknn_lite.inference(inputs=[img_input])
            t1 = time.perf_counter()
            post_process(out, conf_thresh=args.conf)
            t2 = time.perf_counter()
            ti.append((t1-t0)*1000);  tp.append((t2-t1)*1000);  tpl.append((t2-t0)*1000)
        times_infer    = np.array(ti)
        times_post     = np.array(tp)
        times_pipeline = np.array(tpl)
        print('done')

    print_timing_eval(t_preprocess, t_infer, t_post, t_draw,
                      times_infer, times_post, times_pipeline)

    t_total = (time.perf_counter() - t_total_start) * 1000
    print(f'\n总耗时: {t_total:.2f} ms  (含模型加载 {t_load:.2f} ms + 运行时初始化 {t_init:.2f} ms)')

    rknn_lite.release()