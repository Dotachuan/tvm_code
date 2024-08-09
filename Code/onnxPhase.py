import torch
import torch.nn as nn
# import tvm.relay as relay
# import tvm
# import tvm.driver.tvmc as tvmc
# from tvm.contrib import graph_executor

print('run')
bestMo = torch.load("./CFAR_005PID.pth")
bestMo.eval() 
dummy_input = torch.randn(1, 45, 256, 4, device="cuda")
q_label = torch.zeros(size=[256]).cuda()

torch.onnx.export(bestMo, (dummy_input,dummy_input,q_label), "model_1.onnx", opset_version=12, input_names=['radar_input'], output_names=['output']) 

