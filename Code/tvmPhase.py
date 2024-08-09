import onnx
import tvm
from tvm import relay

# Load the ONNX model
onnx_model_path = "model_10.onnx"  # Replace with the path to your model
onnx_model = onnx.load(onnx_model_path)
shape_dict = {"radar_input": (10, 45, 256, 4)}  # Adjust the shape if needed

# Convert the ONNX model to Relay IR
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)