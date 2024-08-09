#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import RadarDataSet
import numpy as np
import os
import time
import onnx
import onnxruntime as ort

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(3407)

def accuracyCount(outputvalue, targetValue):
    outputValue = F.log_softmax(outputvalue, dim=1)
    max_value, max_index = torch.max(outputValue, dim=1)
    acc = max_index == targetValue
    return torch.sum(acc).item(), outputvalue.shape[0]

input_channel = 16
num_class = 23 # Person Number, PID模型需要识别出来的人数(如果涉及到Open-Set问题需要训练模型的时候，这个参数需要修改为Close-set里面的人数)
lossCro = nn.CrossEntropyLoss()
lossMse = nn.MSELoss()
print("float32")
print("One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.")

# Load the test dataset
testset = RadarDataSet("../Records/Test316CFAR005.txt")
testloader = DataLoader(testset, batch_size=128, num_workers=4, pin_memory=True)

# Load the ONNX model
model_path = "model.onnx"
onnx_model = onnx.load(model_path)

# Check the model for any issues
onnx.checker.check_model(onnx_model)

# Create an inference session with GPU support
ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# Warm-up inference
testdata, testlabel = next(iter(testloader))
testdata = testdata.numpy().astype(np.float32)
testlabel = testlabel.numpy().astype(np.float32)

for _ in range(10):
    _ = ort_session.run(None, {'radar_input': testdata})

if __name__ == "__main__":
    acccount = 0
    allcount = 0
    since = time.perf_counter()

    with torch.no_grad():
        for testidx, (testdata, testlabel) in enumerate(testloader):
            testdata = testdata.numpy().astype(np.float32)
            testlabel = testlabel.numpy().astype(np.float32)

            # Run inference with ONNX model
            outputs = ort_session.run(None, {
                'radar_input': testdata,
            })

            # Extract the outputs
            class_output = outputs
            

            # Convert outputs to torch tensors for compatibility with accuracy calculation
            class_output_tensor = torch.tensor(class_output).squeeze(0)
            # print(class_output_tensor.shape)
            testlabel_tensor = torch.tensor(testlabel)
            # print(testlabel_tensor.shape)

            # Calculate accuracy
            accNum, batchNum = accuracyCount(class_output_tensor, testlabel_tensor)
            acccount += accNum
            allcount += batchNum
            print("batch acc: ", accNum/batchNum,testidx)
            if testidx == 54: break
            # print("Testidx: ", testidx, " Class Output: ", class_output)

    time_elapsed = time.perf_counter() - since
    print('Time elapsed is :%sms' % (time_elapsed * 1000))
    accres = acccount / allcount
    print("Accuracy: ", accres)

    # Save the accuracy result
    with open("../Result/CFAR_005PID_ONNX.txt", "a+", encoding="utf-8") as fw:
        fw.write(str(accres))
        fw.write("\n")
