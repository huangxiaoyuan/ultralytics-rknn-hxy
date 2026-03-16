from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
# Load a model
model = YOLO("ultralytics/cfg/models/12/yolov12n-VanillaNet-MSDA.yaml").load("yolo12n.pt")

# Train the model
train_results = model.train(
    data="ultralytics/cfg/datasets/bird_24.yaml",  # path to dataset YAML
    epochs=300,  # number of training epochs
    imgsz=640,  # training image size
    device="0,1"
)

# Evaluate model performance on the validation set
#metrics = model.val()

# Perform object detection on an image
#results = model("./ultralytics/assets/Platalea_minor_004.jpg")
#results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model