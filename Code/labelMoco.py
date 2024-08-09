#encoding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from network import MainModel

class MoCo(nn.Module):
    def __init__(self,input_channel,num_class,base_econder,dim=128,K=8192,m=0.999,T=0.07) -> None:
        '''
        dim: feature dimension(default: 128)  基本编码器输出的结果维度,128是LSTM输出的维度
        K: queue size: number of negative keys K字典队列的大小,默认为4096
        m: moco momentum of update 动量更新参数
        T: 维度参数
        '''
        super(MoCo,self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_econder(input_channel=input_channel,num_class=num_class,num_blocks=1)
        self.encoder_k = base_econder(input_channel=input_channel,num_class=num_class,num_blocks=1)


        for param_q, param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.require_grad = False 

        self.register_buffer("dataqueue",torch.rand(dim,K))
        self.register_buffer("labelqueue",torch.zeros(K,dtype=torch.long))
        self.register_buffer("queue_ptr",torch.zeros(1,dtype=torch.long)) 

    @torch.no_grad()
    def _momentum_update_key_encoder(self): 
        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)  

    @torch.no_grad()
    # def _dequeue_and_enqueue(self, keys, keyslabel):
    #     print("enz")
    #     batch_size = keys.shape[0]
    #     ptr = self.queue_ptr[0]

    #     # Calculate the end index for insertion
    #     end_idx = (ptr + batch_size) % self.K

    #     # --- Create masks for wrap-around handling ---
    #     mask_normal = torch.arange(self.K, device=ptr.device) >= ptr
    #     mask_wrap = torch.arange(self.K, device=ptr.device) < end_idx

    #     # --- Calculate the effective num_remove using a mask ---
    #     mask_remove = torch.logical_or(mask_normal, mask_wrap)  # Combine masks
    #     num_remove = torch.masked_select(torch.arange(self.K, device=ptr.device), mask_remove).size(0)
        
    #     # --- Update dataqueue ---
    #     repeat_factor = torch.div(mask_normal.sum(), keys.shape[0], rounding_mode='floor')
    #     self.dataqueue[:, mask_normal] = keys.T[:, :mask_normal.sum()].repeat(1, repeat_factor)
    #     self.dataqueue[:, mask_wrap] = keys.T[:, mask_normal.sum():]

    #     # --- Update labelqueue ---
    #     self.labelqueue[mask_normal] = keyslabel.float()[:mask_normal.sum()]
    #     self.labelqueue[mask_wrap] = keyslabel.float()[mask_normal.sum():]

    #     # Update the queue pointer
    #     self.queue_ptr[0] = end_idx
    # def _dequeue_and_enqueue(self, keys, keyslabel):
    #     batch_size = keys.shape[0]
    #     ptr = self.queue_ptr[0]
    #     valid_update = (self.K % batch_size == 0)

    #     valid_update = valid_update.to(self.dataqueue.device)

    #     print("Batch size:", batch_size)
    #     print("ptr:", ptr)
    #     print("valid_update:", valid_update)

    #     # Calculate the number of elements to keep and remove from the queue
    #     num_keep = self.K - batch_size
    #     num_remove = batch_size

    #     print("num_keep:", num_keep)
    #     print("num_remove:", num_remove)

    #     # Adjust num_remove if it's larger than the current queue size
    #     if num_remove > ptr:
    #         num_remove = ptr

    #     print("num_remove (after adjustment):", num_remove)

    #     # --- Create masks for wrap-around handling ---
    #     mask_normal = torch.arange(self.K, device=ptr.device) >= ptr
    #     mask_wrap = torch.arange(self.K, device=ptr.device) < (ptr + batch_size) % self.K  # Corrected mask_wrap calculation

    #     print("mask_normal shape:", mask_normal.shape)
    #     print("mask_normal sum:", mask_normal.sum())
    #     print("mask_wrap shape:", mask_wrap.shape)
    #     print("mask_wrap sum:", mask_wrap.sum())

    #     # --- Update dataqueue ---
    #     repeat_factor = torch.div(mask_normal.sum(), keys.shape[0], rounding_mode='floor')
    #     print("repeat_factor:", repeat_factor)

    #     print("keys.T[:, :mask_normal.sum()].shape:", keys.T[:, :mask_normal.sum()].shape)
    #     print("self.dataqueue[:, mask_normal].shape:", self.dataqueue[:, mask_normal].shape)

    #     self.dataqueue[:, mask_normal] = keys.T[:, :mask_normal.sum()].repeat(1, repeat_factor)
    #     self.dataqueue[:, mask_wrap] = keys.T[:, mask_normal.sum():]

    #     # --- Update labelqueue ---
    #     self.labelqueue[mask_normal] = keyslabel.float()[:mask_normal.sum()]
    #     self.labelqueue[mask_wrap] = keyslabel.float()[mask_normal.sum():]

    #     # Update the queue pointer
    #     self.queue_ptr[0] = (ptr + batch_size) % self.K

    #     print("self.queue_ptr[0]:", self.queue_ptr[0])
        
    
    # Working dequeue and enqueue function DO NOT DELETE!!!!!!!!!!!!!!1
    @torch.no_grad()    
    def _dequeue_and_enqueue(self, keys, keyslabel):
        print('test x')
        batch_size = keys.shape[0]
        ptr = self.queue_ptr[0]
        valid_update = (self.K % batch_size == 0)

        valid_update = valid_update.to(self.dataqueue.device)

        # Calculate the number of elements to keep and remove from the queue
        num_keep = self.K - batch_size 
        num_remove = batch_size 

        # Adjust num_remove if it's larger than the current queue size
        if num_remove > ptr:
            num_remove = ptr

        # Update the queue, handling potential size mismatches
        self.dataqueue = torch.cat([self.dataqueue[:, num_remove:], keys.T], dim=1)
        self.labelqueue = torch.cat([self.labelqueue[num_remove:], keyslabel.float()], dim=0)

        # Update the queue pointer
        self.queue_ptr[0] = (ptr + batch_size) % self.K
        
    def forward(self,im_q,im_k,q_label,is_train=False):
        '''
        Input:
            im_q: a batch of query point clouds 
            im_k: a batch of key point clouds 
            q_label: im_q和im_k都具有相同的label
            is_train:
        '''
        cls_q,q = self.encoder_q(im_q) 
        if is_train:
            with torch.no_grad(): 
                self._momentum_update_key_encoder()
                _,k = self.encoder_k(im_k)
                
            queRes = torch.einsum("nc,ck->nk",[q,self.dataqueue.clone().detach()]) 
            quelabel = self.labelqueue.clone().detach() #shape K

            quelabel = torch.stack([quelabel for i in range(queRes.shape[0])],dim=0)

            predictlabel = torch.where(quelabel == q_label.reshape(-1,1),1,-1)
            predictlabel = predictlabel.float()

            self._dequeue_and_enqueue(k,q_label) 

            return cls_q,q,queRes,predictlabel
        else:
            # print("test3", cls_q.shape, cls_q)
            return cls_q
    


if __name__ == "__main__":
    model = MoCo(16,23,MainModel).cuda()
    data = torch.rand(size=[512,45,180,4]).cuda()
    data_k = data.clone()
    q_label = torch.zeros(size=[512]).cuda()
    cls_q,queRes,predictlabel = model(data,data_k,q_label,True)
    print(cls_q.shape)
    lossfunction = nn.MSELoss()
    lossv = lossfunction(queRes,predictlabel)
    lossv.backward()
    print(data.requires_grad)





