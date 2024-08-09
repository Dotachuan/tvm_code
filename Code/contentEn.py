#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.linear1 = nn.Linear(256,50)
        self.act = nn.Softmax(dim=1)
        self.linear2 = nn.Linear(50,1)
    def forward(self,x):
        res = self.linear1(x)
        res = self.act(res)
        res = self.linear2(res)
        return res

class PointAttention(nn.Module):
    def __init__(self):
        super(PointAttention, self).__init__()
        self.linear1 = nn.Linear(256,64)
        self.linear2 = nn.Linear(64,1)
        self.act = nn.Softmax(dim=2)
        self.act2 = nn.ReLU()
    def forward(self,X):
        '''
        :param X:  frame_number, point_number,point_vector
        :return:
        '''
        res = self.linear1(X)
        res = self.act(res)
        res = self.linear2(res)
        #res = self.act2(res)
        #frame, poinnumber, 1
        return res


class PointFeature(nn.Module):
    def __init__(self,input_channel):
        super(PointFeature, self).__init__()
        self.conv1 = nn.Conv1d(input_channel,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,256,1)
        self.pointAttention = PointAttention()
        #self.dropout = nn.Dropout()
    def forward(self,x):
        #x frame,pointnumber,vector
        x = x.transpose(2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(2,1)
        attres = self.pointAttention(x)
        weightRes = attres * x
        #frame, vector
        frameVector = torch.sum(weightRes,dim=1)
        #frameVectorm, _ = torch.max(x,dim=1) 
        #frameVector = torch.cat([frameVector,frameVectorm],dim=1)
        return frameVector
    


if __name__ == "__main__":
    data = torch.tensor([[1,2,3],[3,1,2],[1,3,6]])
    res,_ = torch.max(data,dim=0)
    print(res)