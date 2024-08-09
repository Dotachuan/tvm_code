# import onnx

# # Load your ONNX model
# onnx_model = onnx.load("modified_model.onnx")

# for initializer in onnx_model.graph.initializer:
#     if initializer.data_type == onnx.TensorProto.INT64:
#         print("INT64 Initializer:", initializer.name)

# # Iterate through nodes and their attributes
# for node in onnx_model.graph.node:
#     for attribute in node.attribute:
#         if attribute.type == onnx.AttributeProto.INTS and any(i > 2**31 - 1 for i in attribute.ints):
#             print("INT64 Attribute:", attribute.name, "in Node:", node.name)
import onnx
import numpy as np

model_path = "model_10.onnx"
modified_model_path = "modified_model.onnx"

# Load the model
model = onnx.load(model_path)

# Find and modify the INT64 initializer 
for initializer in model.graph.initializer:
    if initializer.data_type == onnx.TensorProto.INT64:
        print("Data type before conversion:", initializer.data_type)  

        # 1. Convert the tensor data to int32
        int32_data = np.frombuffer(initializer.raw_data, dtype=np.int64).astype(np.int32)

        # 2. Update the data type and raw data 
        initializer.data_type = onnx.TensorProto.INT32
        initializer.raw_data = int32_data.tobytes()

        print("Data type after conversion:", initializer.data_type) 

# Save the modified model
onnx.save(model, modified_model_path)