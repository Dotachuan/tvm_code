# encoding=utf-8

import torch
import torch.nn as nn
from labelMoco import MoCo
from network import MainModel, PointFeature, LSTransBlock

# Instantiate the sub-modules 
input_channel = 16
num_class = 23 
base_encoder = MainModel(input_channel, num_class, num_blocks=1)
moco_model = MoCo(input_channel, num_class, base_encoder).cuda()

points = torch.rand(size=[1, 45, 256, 4]).cuda()
feature = torch.rand(size=[1, 45, 256]).cuda()
data = torch.rand(size=[1, 45, 256, 4]).cuda() 
data_k = data.clone()
label = torch.zeros(size=[1]).cuda()
is_train_tensor = torch.tensor([True]).cuda()

# 1. Trace PointFeature
traced_point_feature = torch.jit.trace(moco_model.encoder_q.ContentEncoder, points) 
traced_point_feature.save("traced_point_feature.pt")

# 2. Trace LSTransBlock
traced_lstm_trans_block = torch.jit.trace(moco_model.encoder_q.LSTransBlock1, feature)
traced_lstm_trans_block.save("traced_lstm_trans_block.pt")

# 3. Trace MainModel 
traced_main_model = torch.jit.trace(moco_model.encoder_q, points)
traced_main_model.save("traced_main_model.pt")

# 4. Trace MoCo (modified forward for tracing)
class MoCoTraced(nn.Module):
    def __init__(self, moco_model):
        super(MoCoTraced, self).__init__()
        self.encoder_q = moco_model.encoder_q
        self.encoder_k = moco_model.encoder_k
        self.K = moco_model.K
        self.m = moco_model.m
        self.T = moco_model.T
        self.register_buffer("dataqueue", moco_model.dataqueue)
        self.register_buffer("labelqueue", moco_model.labelqueue)
        self.register_buffer("queue_ptr", moco_model.queue_ptr)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keyslabel):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.dataqueue[:, ptr:ptr + batch_size] = keys.T
        self.labelqueue[ptr:ptr + batch_size] = keyslabel
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, q_label, is_train):
        cls_q, q = self.encoder_q(im_q)
        if is_train:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                _, k = self.encoder_k(im_k)
            queRes = torch.einsum("nc,ck->nk", [q, self.dataqueue.clone().detach()])
            quelabel = self.labelqueue.clone().detach()
            quelabel = torch.stack([quelabel for i in range(queRes.shape[0])], dim=0)
            predictlabel = torch.where(quelabel == q_label.reshape(-1, 1), 1, -1)
            predictlabel = predictlabel.float()
            self._dequeue_and_enqueue(k, q_label)
            return cls_q, q, queRes, predictlabel
        else:
            return cls_q 

traced_moco = torch.jit.trace(MoCoTraced(moco_model), (data, data_k, label, is_train_tensor))
traced_moco.save("traced_moco.pt")

print("All modules successfully traced and saved!") 