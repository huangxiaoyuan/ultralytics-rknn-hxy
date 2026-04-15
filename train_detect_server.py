from click.core import batch

from ultralytics import YOLO
import warnings
import os
#os.environ['MKL_THREADING_LAYER'] = 'GNU'  # ← 必须第一行
#os.environ['NCCL_TIMEOUT'] = '300'          # ← 超时从3小时缩短到5分钟
#warnings.filterwarnings('ignore')
#local_rank = int(os.environ.get("LOCAL_RANK", 0))
# Load a model
if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml", task="detect").load("yolo8n.pt")
    train_results = model.train(
        data="ultralytics/cfg/datasets/bird_cub200_server.yaml",
        epochs=300,
        imgsz=640,
        device="0,1",
        batch=16,
        val=True,
    )
    # 自动获取本次训练的保存目录
    best_pt = "weights/best.pt"
    print(f"使用模型: {best_pt}")
    # 单卡评估
    metrics = YOLO(str(best_pt)).val(
        data="ultralytics/cfg/datasets/bird_cub200_server.yaml",
        device=0,
    )
    print(metrics)

# Evaluate model performance on the validation set
#metrics = model.val()

# Perform object detection on an image
#results = model("./ultralytics/assets/Platalea_minor_004.jpg")
#results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model