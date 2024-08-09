import torch
import torch.onnx

# Load your PyTorch model
bestMo = torch.load("resulttransMoco316CFAR002.pth")
bestMo.eval()

# Create two dummy inputs for 'im_q' and 'im_k' (since your MoCo class expects them)
dummy_input_q = torch.randn(256, 45, 256, 4, device='cuda')
dummy_input_k = torch.randn(256, 45, 256, 4, device='cuda')

# Create a dummy label (this is not necessary for inference, but you can include it if you want)
dummy_label = torch.randint(0, 23, (256,), device='cuda')  # 23 is the number of classes

# Export the model to ONNX, feeding both dummy inputs
torch.onnx.export(
    bestMo,  # Your PyTorch model
    (dummy_input_q, dummy_input_k, dummy_label),  # Tuple of dummy inputs
    "model.onnx",  # Name of the ONNX file to save
    opset_version=11,  # ONNX operator set version (adjust if needed)
    export_params=True,  # Include model parameters in the ONNX file
    input_names=["radar_input_q", "radar_input_k", "label"],  # Optional: Specify input names
    output_names=["class_output", "hidden_output"],  # Optional: Specify output names
)

print("ONNX model exported successfully!")