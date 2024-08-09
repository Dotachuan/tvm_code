from dataloader import RadarDataSet
from torch.utils.data import DataLoader

testset = RadarDataSet("../Records/Test316CFAR005.txt")
testloader = DataLoader(testset, batch_size=10, num_workers=4, pin_memory=True)
import tvm
from tvm.contrib import graph_executor
import numpy as np
import torch
import torch.nn.functional as F
import time

loaded_lib = tvm.runtime.load_module("compiled_model.so")
with open("compiled_model.json", "r") as f:
    loaded_json = f.read()
with open("compiled_model.params", "rb") as f:
    loaded_params = bytearray(f.read())

# Create graph executor
ctx = tvm.gpu(0)  
module = graph_executor.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)

def accuracyCount(outputvalue, targetValue):
    outputValue = F.log_softmax(outputvalue, dim=1)
    max_value, max_index = torch.max(outputValue, dim=1)
    acc = max_index == targetValue
    return torch.sum(acc).item(), outputvalue.shape[0]

acccount = 0
allcount = 0
since = time.perf_counter()

with torch.no_grad():
    for testidx, (testdata, testlabel) in enumerate(testloader):
        testdata = testdata.numpy().astype(np.float32)
        testlabel = testlabel.numpy().astype(np.float32)

        module.set_input('radar_input', testdata)
        module.run()
        
        outputs = [module.get_output(i).asnumpy() for i in range(module.get_num_outputs())]

        class_output_tensor = torch.tensor(outputs)
        testlabel_tensor = torch.tensor(testlabel)
        # Calculate accuracy
        accNum, batchNum = accuracyCount(class_output_tensor, testlabel_tensor)
        acccount += accNum
        allcount += batchNum
        print("batch acc: ", accNum / batchNum)
        if testidx == 10: break

time_elapsed = time.perf_counter() - since
print('Time elapsed is :%sms' % (time_elapsed * 1000))
accres = acccount / allcount
print("Accuracy: ", accres)

# Save the accuracy result
with open("../Result/CFAR_005PID_TVM.txt", "a+", encoding="utf-8") as fw:
    fw.write(str(accres))
    fw.write("\n")