from sympy import false

from ultralytics import YOLO


model = YOLO("E:/bird/yolo11_run/server-119/train5/weights/best.pt")
model.export(
    format="onnx",
    imgsz=320,
    dynamic= False,  # 动态batch和分辨率
    simplify= False,  # 应用图优化
    half = True,
    opset=13,  # 使用较新算子集
    name="best_yolo12_320-fp16.onnx")