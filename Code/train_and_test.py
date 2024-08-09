#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from labelMoco import MoCo
from network import MainModel
from torch.utils.data import DataLoader
from dataloader import RadarDataSet
import numpy as np
import torch.nn.functional as F
import os
import torch.optim as optim
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


torch.manual_seed(3407)
def accuracyFunction(outputvalue,targetValue):
    outputValue = F.log_softmax(outputvalue,dim=1)
    max_value,max_index = torch.max(outputValue,dim=1)
    acc = max_index == targetValue
    acc = torch.sum(acc)/acc.shape[0]               
    return acc.item()               #单个张量转为标量

def accuracyCount(outputvalue,targetValue):
    outputValue = F.log_softmax(outputvalue, dim=1)
    max_value, max_index = torch.max(outputValue, dim=1)
    acc = max_index == targetValue
    #acc = torch.sum(acc) / acc.shape[0]
    #confusionMatrix = np.zeros(shape=[outputvalue.shape[1],outputvalue.shape[1]])
    #maxindexList = max_index.tolist()
    #targetList = targetValue.tolist()
    #acclist = acc.tolist()
    #filenames = []
    #for ii in range(len(targetList)):
    #    confusionMatrix[targetList[ii],maxindexList[ii]] += 1
    #for ii in range(len(acclist)):
     #   if acclist[ii] == False:
     #       filenames.append([testSegs[0][ii],testSegs[1][ii]])
    return torch.sum(acc).item(),outputvalue.shape[0]

input_channel = 16
num_class = 23 #Person Number, PID模型需要识别出来的人数(如果涉及到Open-Set问题需要训练模型的时候，这个参数需要修改为Close-set里面的人数)
#model = MoCo(input_channel,num_class,MainModel).cuda()
lossCro = nn.CrossEntropyLoss()
lossMse = nn.MSELoss()
#lossMse = nn.L1Loss()

model = torch.load("CFAR_005PID.pth")

trainset = RadarDataSet("../Records/Train316CFAR005.txt")
testset = RadarDataSet("../Records/Test316CFAR005.txt")
trainloader = DataLoader(trainset,batch_size=128,num_workers=4,pin_memory=True)
testloader = DataLoader(testset,batch_size=128,num_workers=4,pin_memory=True)

lr = 0.001
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
epoes = 350
bestacc = 0.0



if __name__ == "__main__":
    
    # for epo in range(1):
    # #     model.train()
    # #     for trainidx,(traindata,trainlabel) in enumerate(trainloader):
    # #         # print(traindata.shape)
    # #         if trainidx == 423:
    # #             #traindata = torch.cat([traindata,traindata[:512-traindata.shape[0]].detach()],dim=0)
    # #             #trainlabel = torch.cat([trainlabel,trainlabel[:512-traindata.shape[0]].detach()])
    # #             continue

    # #         traindata = traindata.cuda().float()
    # #         print(traindata.shape)
    # #         trainlabel = trainlabel.cuda()
    # #         print(trainlabel.shape)
    # #         #print("----------------load data successfully-------------------------")
    # #         optimizer.zero_grad()
    # #         claTrain,_,queTrain,queLabel = model(traindata,traindata,trainlabel,True)
    # #         #print("------------------obtain result------------------------------")
    # #         trainlossCro = lossCro(claTrain,trainlabel)
    # #         trainlossMse = lossMse(queTrain,queLabel)
    # #         trainloss = trainlossCro + 0.1 * trainlossMse
    # #         trainloss.backward()
    # #         optimizer.step()
            
    # #         trainacc = accuracyFunction(claTrain,trainlabel)
    # #         print("Epoch: ",epo," trainidx: ",trainidx,
    # #               " trainlossCro: ",trainlossCro.data.item(),
    # #               " trainlossMse: ",trainlossMse.data.item(),
    # #               " acc: ",trainacc)
    # #     # 测试
    with torch.no_grad():
        model.eval()
        acccount = 0
        allcount = 0
        
        since = time.perf_counter()
        for testidx,(testdata,testlabel) in enumerate(testloader):
            testdata = testdata.cuda().float()
            testlabel = testlabel.cuda()
            testres = model(testdata,None,None,False)
            testlossCro = lossCro(testres,testlabel)
            accNum,batchNum = accuracyCount(testres,testlabel)
            acccount += accNum
            allcount += batchNum
            print("batch acc: ", accNum/batchNum,testidx)
            #if testidx == 10: break
            # print(" testidx: ",testidx," testloss: ",testlossCro.data.item())
        time_elapsed = time.perf_counter() - since
        print('Time elapsed is :%sms' %(time_elapsed*1000))
        accres = acccount / allcount
        print("Accuracy: ",accres)
        
        fw = open("../Result/CFAR_005PID.txt","a+",encoding="utf-8") 
        fw.write(str(accres))
        fw.write("\n")
        fw.close()
        if accres > bestacc:
            bestacc = accres
            #注意：这个位置涉及可能会涉及如何保存模型的方式，我之前的实验为了方便就把结构和参数都保存了
            torch.save(model,"../Result/CFAR_005PID.pth")  
   
    # torch.save(model,"./CFAR_005PID.pth")
            
'''
    #load一下测试结果
    #可视化测试，边缘计算里面用不到
    bestMo = torch.load("resulttransMoco.pth")
    fr = open("../Records/Test.txt","r",encoding="utf-8")
    allLines = fr.readlines()
    fr.close()
    #print(bestMo)
    
    fw = open("../ResultView/error.txt","w",encoding="utf-8")
    with torch.no_grad():
        for testidx,(testdata,testlabel) in enumerate(testloader):
            print(testidx)
            testdata = testdata.cuda().float()
            testlabel = testlabel.cuda()
            testres = bestMo(testdata,None,None,False)
            testres = F.log_softmax(testres,dim=1)
            max_value,max_index = torch.max(testres,dim=1)
            
            if max_index != testlabel:
                Segments = allLines[testidx*45:testidx*45+45]
                for line in Segments:
                    line = line.strip().split("\t")[0]
                    fw.write(line)
                    fw.write("\t")
                    fw.write(str(max_index.data.item()))
                    fw.write("\n")
                fw.write("\n")
            #torch.save(testres,"../ResultView/"+str(testidx))
            #print(testidx)
        fw.close()
'''
    
    