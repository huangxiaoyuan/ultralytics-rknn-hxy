from ultralytics import YOLO
import warnings
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'  # ← 必须第一行
os.environ['NCCL_TIMEOUT'] = '300'          # ← 超时从3小时缩短到5分钟
warnings.filterwarnings('ignore')

# Load a model
model = YOLO("ultralytics/cfg/models/12/yolo12n_npu_h.yaml",task="detect").load("yolo12n.pt")

# Train the model
train_results = model.train(
    data="ultralytics/cfg/datasets/bird_24_server.yaml",  # path to dataset YAML
    epochs=300,  # number of training epochs
    imgsz=640,  # training image size
    device="0,1"  # distributed training must setting
)

# Evaluate model performance on the validation set
#metrics = model.val()

# Perform object detection on an image
#results = model("./ultralytics/assets/Platalea_minor_004.jpg")
#results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model