import torch
from network import MainModel
import tvm
from tvm import relay
import tvm.relay as relay
import tvm
#import tvm.driver.tvmc as tvmc
#from tvm.contrib import graph_executor
#import numpy as np

# 1. Load and Prepare Model
#bestMo = torch.load("CFAR_005PID.pth")
bestMo = torch.load("resulttransMoco316CFAR002.pth")
bestMo.eval() 
dummy_input = torch.randn(128, 45, 256, 4).cuda()
q_label = torch.zeros(size=[512]).cuda()
# 2. ONNX Export

scripted_model = torch.jit.trace(bestMo, (dummy_input,dummy_input,q_label)).eval()
input_name = "input0"
shape_list = [(input_name, dummy_input.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

#result = scripted_model(dummy_input)
#print(result.shape)
# input_name = "input0"
# shape_list = [(input_name, dummy_input.shape)]
# mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
#torch.onnx.export(scripted_model ,dummy_input,"model.onnx",opset_version=11)
#torch.onnx.export(bestMo, dummy_input, "model.onnx", opset_version=11, export_params=True,)
# input_names=['radar_input'], output_names=['class_output', 'hidden_output'] 


