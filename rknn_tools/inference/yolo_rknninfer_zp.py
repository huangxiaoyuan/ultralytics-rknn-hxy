import os
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# ─── 配置 ────────────────────────────────────────────────────────
RK3588_RKNN_MODEL = 'bird_rknn_model/320/best_bird_320_op13-331pruning.rknn'
IMG_PATH = 'Hydrophasianus_chirurgus_172.jpg'
OBJ_THRESH = 0.5
NMS_THRESH = 0.45
IMG_SIZE = 320  # 必须与模型输入一致

CLASSES = ("accipiter_nisus", "arenaria_interpres", "calidris_falcinellus", "calidris_tenuirostris",
           "calliope_calliope", "centropus_sinensis", "circus_spilonotus", "egetta_eulophotes",
           "egretta_sacra", "elanus_caeruleus", "falco_amurensis", "falco_tinnunculus",
           "garrulax_canorus", "halcyon_smyrnensis", "hydrophasianus_chirurgus", "leiothrix_argentauris",
           "leiothrix_lutea", "limnodromus_semipalmatus", "merops_philippinus", "milvus_migrans",
           "numenius_arquata", "pandion_haliaetus", "platalea_leucorodia", "platalea_minor")


# ─── 后处理函数 ──────────────────────────────────────────────────
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes, scores):
    x, y = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0];
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]]);
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]]);
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1e-5);
        h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= NMS_THRESH)[0] + 1]
    return np.array(keep)


def post_process(outputs, conf_thresh=OBJ_THRESH):
    pred = outputs[0][0].transpose(1, 0)  # (2100, 28)
    class_scores = 1 / (1 + np.exp(-pred[:, 4:]))  # Sigmoid
    class_max_score = np.max(class_scores, axis=-1)
    classes = np.argmax(class_scores, axis=-1)

    mask = class_max_score > conf_thresh
    if not np.any(mask): return None, None, None

    boxes = xywh2xyxy(pred[mask, :4])
    scores = class_max_score[mask]
    cls_indices = classes[mask]

    nboxes, nclasses, nscores = [], [], []
    for c in set(cls_indices):
        inds = np.where(cls_indices == c)
        b, s = boxes[inds], scores[inds]
        keep = nms_boxes(b, s)
        nboxes.append(b[keep]);
        nclasses.append(cls_indices[inds][keep]);
        nscores.append(s[keep])

    if not nboxes: return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


# ─── 主程序 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    rknn_lite = RKNNLite()

    # 1. 加载模型
    if rknn_lite.load_rknn(RK3588_RKNN_MODEL) != 0:
        print('Load model failed');
        exit()

    # 2. 初始化 Runtime 并指定核心
    # RK3588 有三个 NPU 核心，可以绑定其中一个以提高多进程效率
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime failed');
        exit()

    # 3. 预处理 (优化点：保持 uint8，不要在 Python 里转 float)
    img_src = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # 增加维度 (1, 320, 320, 3)
    img_input = np.expand_dims(img_input, 0)

    # 4. 推理
    print('--> Running inference')
    t0 = time.time()

    # 在 RKNNLite 中，inference 依然是最快的 Python 接口
    # 确保输入是 uint8，且 shape 为 NHWC，这样 RKNN 驱动内部不会做额外的预处理
    outputs = rknn_lite.inference(inputs=[img_input])

    print('Inference time: {:.2f} ms'.format((time.time() - t0) * 1000))

    # 5. 后处理
    boxes, classes, scores = post_process(outputs)

    # 6. 画图
    if boxes is not None:
        for box, score, cl in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_src, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_src, f'{CLASSES[cl]} {score:.2f}', (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite('result_optimized.jpg', img_src)
        print('Saved to result_optimized.jpg')
    else:
        print('No objects detected.')

    rknn_lite.release()