import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
from dataloader import RadarDataSet
import onnx
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import time
import torch.nn as nn
import torch.nn.functional as F
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

def accuracyCount(outputvalue, targetValue):
    outputValue = F.log_softmax(outputvalue, dim=1)
    max_value, max_index = torch.max(outputValue, dim=1)
    acc = max_index == targetValue
    return torch.sum(acc).item(), outputvalue.shape[0]

trainset = RadarDataSet("../Records/Train316CFAR005.txt")
testset = RadarDataSet("../Records/Test316CFAR005.txt")
trainloader = DataLoader(trainset,batch_size=128,num_workers=4,pin_memory=True)
testloader = DataLoader(testset,batch_size=128,num_workers=4,pin_memory=True)

epoes = 350

model = torch.load("CFAR_005PID.pth")
onnx_model = onnx.load('model.onnx') 

# for epo in range(epoes):
#     with torch.no_grad():
#         model.eval()
#         acccount = 0
#         allcount = 0
            
#         for testidx,(testdata,testlabel) in enumerate(testloader):
#             testdata = testdata.cuda().float()
#             testlabel = testlabel.cuda()
#             testres = model(testdata,None,None,False)
#             print(testdata.shape)
#             testlossCro = lossCro(testres,testlabel)
#             accNum,batchNum = accuracyCount(testres,testlabel)
#             acccount += accNum
#             allcount += batchNum
            
#             print("Epoch: ",epo," testidx: ",testidx," testloss: ",testlossCro.data.item())
    
#             accres = acccount / allcount

#             print("Accuracy: ",accres)
            
#             fw = open("../Result/CFAR_005PID.txt","a+",encoding="utf-8") 
#             fw.write(str(accres))
#             fw.write("\n")
#             fw.close()
#             if accres > bestacc:
#                 bestacc = accres
#                 #注意：这个位置涉及可能会涉及如何保存模型的方式，我之前的实验为了方便就把结构和参数都保存了
#                 torch.save(model,"../Result/CFAR_005PID.pth")  
   
#     # torch.save(model,"./CFAR_005PID.pth")
target = 'llvm'
dummy_input = np.random.rand(128,45,256,4).astype(np.float32)
dev = tvm.device(str(target),0)
print(dummy_input.dtype)
input_name = 'radar_input'  # 注意这里为之前导出onnx模型中的模型的输入id，这里为0
shape_dict = {input_name: dummy_input.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

dtype = 'float32'


number = 10
repeat = 1
min_repeat_ms = 0  # 调优 CPU 时设置为 0
timeout = 10  # 秒


runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "test.json",
}

tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="reg")
    
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],)

with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

module = graph_executor.GraphModule(lib["default"](dev))
acccount = 0
allcount = 0

since = time.perf_counter()
for testidx,(testdata,testlabel) in enumerate(testloader):
    print("batch :",testidx)
    module.set_input(input_name, testdata)
    module.run()
    tvm_output = torch.from_numpy(module.get_output(0).numpy())
    accNum,batchNum = accuracyCount(tvm_output,testlabel)
    acccount += accNum
    allcount += batchNum
    if testidx == 54: break
    print("batch acc: ", accNum/batchNum,testidx)

time_elapsed = time.perf_counter() - since
print('Time elapsed is :%sms' %(time_elapsed*1000))
accres = acccount / allcount
print("Accuracy: ",accres)




#model(input)
# session = ort.InferenceSession("model.onnx")

# input_name = session.get_inputs()[0].name
# input_data = input.numpy()

# outputs = session.run(None, {input_name: input_data})
