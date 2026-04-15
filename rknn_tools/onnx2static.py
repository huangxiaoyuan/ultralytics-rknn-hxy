import onnx
from onnx.tools import update_model_dims

model = onnx.load("best_yolov12n_320.onnx")

# Fix input shape
for input in model.graph.input:
    if input.name == "images":
        d = input.type.tensor_type.shape.dim
        d[0].dim_value = 1    # batch
        d[1].dim_value = 3    # channels
        d[2].dim_value = 320  # height
        d[3].dim_value = 320  # width

onnx.save(model, "best_yolov12n_320_fixed.onnx")