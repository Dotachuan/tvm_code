import torch
from network import MainModel
from labelMoco import MoCo
model_state_dict = torch.load("resulttransMoco316CFAR002.pth") 

# Create a new instance of your MoCo model 
input_channel = 16
num_class = 23
model = MoCo(input_channel, num_class, MainModel).cuda()

# Load the state dictionary into the encoder_q
model.encoder_q.load_state_dict(model_state_dict['encoder_q']) # Assuming your saved state_dict has a key 'encoder_q'

model.eval() 

batch_size = 1 
frames = 45
points = 180
features = 4
dummy_input = torch.randn(batch_size, frames, points, features).cuda()

 
torch.onnx.export(
    model.encoder_q,                 
    dummy_input,                      
    "radar_model.onnx",             
    export_params=True,             
    opset_version=11,               
    do_constant_folding=True,       
    input_names=['input_data'],       
    output_names=['output_scores'],   
    dynamic_axes={
        'input_data': {0: 'batch_size'},  
        'output_scores': {0: 'batch_size'}
    }
)