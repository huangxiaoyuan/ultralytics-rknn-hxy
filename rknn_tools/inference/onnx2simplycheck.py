import onnxruntime as ort
import numpy as np

dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

for path in ["E:/bird/yolo_npu/train2/weights/best_640.onnx", "E:/bird/yolo_npu/train2/weights/export/best_640_op13.onnx"]:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})
    print(f"{path}")
    print(f"  output shape: {out[0].shape}")
    print(f"  output sample: {out[0][0, :5, :5]}")  # 打印前几个值对比