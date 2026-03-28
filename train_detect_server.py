from click.core import batch
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/12/yolo12n.yaml", task="detect").load("yolo12n.pt")
    train_results = model.train(
        data="ultralytics/cfg/datasets/bird_cub200_server.yaml",
        epochs=300,
        imgsz=640,
        device="0,1",
        batch=16,
        #val=False,
    )
    # 自动获取本次训练的保存目录


# Evaluate model performance on the validation set
#metrics = model.val()

# Perform object detection on an image
#results = model("./ultralytics/assets/Platalea_minor_004.jpg")
#results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model