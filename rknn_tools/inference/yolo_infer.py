import os
import time
import sys
import numpy as np
import cv2
import platform
from rknnlite.api import RKNNLite

# ─── 模型路径配置 ────────────────────────────────────────────────
RK3566_RK3568_RKNN_MODEL = 'yolov5s_for_rk3566_rk3568.rknn'
RK3588_RKNN_MODEL        = 'bird_rknn_model/yolov12-bird-320_rk3588.rknn'
RK3562_RKNN_MODEL        = 'yolov5s_for_rk3562.rknn'
IMG_PATH                 = 'Hydrophasianus_chirurgus_172.jpg'

# ─── 推理参数 ────────────────────────────────────────────────────
OBJ_THRESH = 0.5      # 置信度阈值（严格大于，过滤 sigmoid(0)=0.5 的噪声）
NMS_THRESH = 0.45     # NMS IoU 阈值
IMG_SIZE   = 320      # 输入分辨率

# ─── 类别列表 ────────────────────────────────────────────────────
CLASSES = (
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
)

DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'


# ─── 设备检测 ────────────────────────────────────────────────────
def get_host():
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                elif 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host


# ─── 工具函数 ────────────────────────────────────────────────────
def xywh2xyxy(x):
    """将 [cx, cy, w, h] 转换为 [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes, scores):
    """非极大值抑制"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


# ─── YOLOv12 单头后处理 ──────────────────────────────────────────
def yolov12_post_process(output, conf_thresh=OBJ_THRESH, nms_thresh=NMS_THRESH):
    """
    输入: output[0] shape = (1, 28, 2100)
      - 4 个坐标 (cx, cy, w, h)
      - 24 个类别 logits
    输出: boxes (xyxy), classes, scores
    """
    pred = output[0][0]               # (28, 2100)
    pred = pred.transpose(1, 0)       # (2100, 28)

    boxes_xywh  = pred[:, :4]
    class_logits = pred[:, 4:]
    class_scores = 1 / (1 + np.exp(-class_logits))  # sigmoid

    class_max_score = np.max(class_scores, axis=-1)
    classes = np.argmax(class_scores, axis=-1)

    # 严格大于阈值，过滤 sigmoid(0)=0.5 的量化噪声
    mask = class_max_score > conf_thresh
    boxes_xywh      = boxes_xywh[mask]
    class_max_score = class_max_score[mask]
    classes         = classes[mask]

    if len(boxes_xywh) == 0:
        return None, None, None

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes_xyxy[inds]
        s = class_max_score[inds]
        keep = nms_boxes(b, s)
        nboxes.append(b[keep])
        nclasses.append(classes[inds][keep])
        nscores.append(s[keep])

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


# ─── 绘制结果 ────────────────────────────────────────────────────
def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        print('class: {}, score: {:.4f}'.format(CLASSES[cl], score))
        print('box x1,y1,x2,y2: [{}, {}, {}, {}]'.format(x1, y1, x2, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, '{} {:.2f}'.format(CLASSES[cl], score),
                    (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ─── 主程序 ──────────────────────────────────────────────────────
if __name__ == '__main__':

    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite()

    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 图像预处理
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)   # (1, 320, 320, 3) NHWC

    # 推理
    print('--> Running model')
    t0 = time.time()
    outputs = rknn_lite.inference(inputs=[img])
    print('Inference time: {:.1f} ms'.format((time.time() - t0) * 1000))
    print('done')

    # 后处理
    boxes, classes, scores = yolov12_post_process(outputs)

    # 绘制并保存
    img_out = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_out, boxes, scores, classes)
    else:
        print('No objects detected.')

    cv2.imwrite('result.jpg', img_out)
    print('Result saved to result.jpg')

    rknn_lite.release()