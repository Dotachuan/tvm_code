import torch
import time

dummy_input = torch.randn(1,45,180,4).cuda()
model = torch.load("resulttransMoco316CFAR002.pth")
q_label = torch.zeros(size=[1]).cuda()  


since = time.perf_counter()
for i in range(100):
    print(i)
    model(dummy_input,dummy_input,q_label,False)
time_elapsed = time.perf_counter() - since
print('Time elapsed is :%sms' %(time_elapsed*1000))

