from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("/data/rknn/runs/detect/train/weights/best.pt")
    # model = YOLO("/data/yolov11/runs/detect/train47/weights/best.pt")
    # Evaluate model performance on the validation set
    metrics = model.val(data="ultralytics/cfg/datasets/bird_cub200_server.yaml")

    # Perform object detection on an image
    # results = model("./ultralytics/assets/Platalea_minor_004.jpg")
    # results[0].show()

    # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model
