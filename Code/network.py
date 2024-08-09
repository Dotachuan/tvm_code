#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from contentEn import PointFeature
from PointPositionEn import getAugResult
import math



class Self_Attention(nn.Module):
    def __init__(self,input_size,hidden_size,num_heads) -> None:
        super(Self_Attention,self).__init__()
        self.attention_head_size = int(hidden_size/num_heads)
        self.all_head_size = hidden_size
        self.num_heads = num_heads
        self.key_layer = nn.Linear(input_size,hidden_size)
        self.query_layer = nn.Linear(input_size,hidden_size)
        self.value_layer = nn.Linear(input_size,hidden_size)

    def forward(self,X):
        '''
        Input:
            X: batchsize, frame, input_size
        Output:
            key: batchsize, frame, hidden_size
            query: batchsize, frame, hidden_size
            value: batchsize, frame, hidden_size
        '''
        key,query,value = self.key_layer(X),self.query_layer(X),self.value_layer(X)
        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attentionScores = torch.matmul(query_heads,key_heads.permute(0,1,3,2))
        attentionScores = attentionScores / math.sqrt(self.attention_head_size)
        attentionProbs = F.softmax(attentionScores,dim=-1)
        context = torch.matmul(attentionProbs,value_heads) #batchsize, num_heads,frame,attention_head_size
        context = context.permute(0,2,1,3).contiguous()
        new_size = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_size)

        return context

    def trans_to_multiple_heads(self,X):
        '''
        X: batchsize, frame, hidden_size
        '''
        new_size = X.size()[:-1] + (self.num_heads,self.attention_head_size) #batchsize, frame, num_heads, attention_head_size
        X = X.view(new_size)
        return X.permute(0,2,1,3) #batchsize, num_heads,frame,attention_head_size




class LSTMMeanModel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LSTMMeanModel, self).__init__()
        self.BiLSTM = nn.LSTM(input_size,hidden_size,num_layers=1,batch_first=True,bidirectional=True)
    def forward(self,X):
        self.BiLSTM.flatten_parameters()
        outdata,(hidden,cells) = self.BiLSTM(X)
        hiddens = hidden.transpose(1,0)
        hiddens = torch.sum(hiddens,dim=1)
        return hiddens

class LSTransBlock(nn.Module):
    def __init__(self,d_model,n_heads) -> None:
        super(LSTransBlock,self).__init__()
        self.translayer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
                                                     batch_first=True)
        #self.selfattentionlayer = Self_Attention(input_size=d_model,hidden_size=d_model,num_heads=n_heads)
        self.BiLSTM = nn.LSTM(d_model,d_model//2,num_layers=1,batch_first=True,bidirectional=True)

    def forward(self,X):
        '''
        X : BatchSize, FrameNumber, Feature
        '''
        self.BiLSTM.flatten_parameters()
        res = self.translayer(X)

        outdata,(hidden,cells) = self.BiLSTM(res)
        hidden = torch.sum(hidden,dim=0)
        
        return outdata,hidden



class MainModel(nn.Module):
    def __init__(self,input_channel,num_class,num_blocks):
        super(MainModel,self).__init__()
        self.ContentEncoder = PointFeature(input_channel=input_channel)
        self.LSTransBlock1 = LSTransBlock(d_model=256,n_heads=4)
        #self.LSTransBlock2 = LSTransBlock(d_model=256,n_heads=4)
        #self.LSTransBlock3 = LSTransBlock(d_model=256,n_heads=4)
        self.predictLayers = nn.Sequential(nn.Linear(128*num_blocks,64),nn.ReLU(inplace=True),
                                           nn.Linear(64,num_class))
    # commented some stuff and passed dummy data debugging 
    def forward(self,Points):
        '''
        Points: BatchSize,45,180,4
        '''
        PointsY = Points[:,:,:,1]
        mindata, _ = torch.min(PointsY,dim=2)
        mindata, _ = torch.min(mindata,dim=1)
        
        
        Points[:,:,:,1] = Points[:,:,:,1] - mindata.reshape(-1,1,1)
        BatchRes = []
        #BatchResO = []
        for i in range(Points.shape[0]):
            AugPoints = getAugResult(Points[i])
            Feature = self.ContentEncoder(AugPoints)
            #FeatureO = self.ContentEncoder(AugPointsO)
            #BatchResO.append(FeatureO)
            BatchRes.append(Feature)
        BatchRes = torch.stack(BatchRes,dim=0)
        

        #BatchResO = torch.stack(BatchResO,dim=0)

        outres1,hiddenres1 = self.LSTransBlock1(BatchRes)
        #outres2,hiddenres2 = self.LSTransBlock2(BatchResO)
        #hiddenres = hiddenres1 + hiddenres2

        #_,hiddenres3 = self.LSTransBlock3(outres2)
        #hiddenres = torch.cat([hiddenres1,hiddenres2],dim=1)
        clres = self.predictLayers(hiddenres1)
        print("test2")
        return clres,hiddenres1
        # return torch.randn(256, 23), torch.randn(128) # Return dummy
        
        


if __name__ == "__main__":
    mo = MainModel(input_channel=16,num_class=23,num_blocks=1).cuda()
    print(mo)
    data = torch.rand(size=[512,45,256,4]).cuda()
    clres,_ = mo(data)
    print(clres.shape)