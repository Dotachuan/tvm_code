# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather
import math


class PointAttention(nn.Module):
    def __init__(self):
        super(PointAttention, self).__init__()
        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)
        self.act = nn.Softmax(dim=2)

    def forward(self, X):
        """
        :param X:  frame_number, point_number, point_vector
        :return:
        """
        res = self.linear1(X)
        res = self.act(res)
        res = self.linear2(res)
        return res


class PointFeature(nn.Module):
    def __init__(self, input_channel):
        super(PointFeature, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.pointAttention = PointAttention()

    def forward(self, x):
        # x frame, pointnumber, vector
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(2, 1)
        attres = self.pointAttention(x)
        weightRes = attres * x
        # frame, vector
        frameVector = torch.sum(weightRes, dim=1)
        return frameVector


def getKnnRes(pointsAnchor, pointsLeft, pointsRight, K=3):
    """
    input:
        points1: BatchSize, PointsNumber, Dimension(4) Anchor
        points2: BatchSize, PointsNumber, Dimension(4) Left Source
        points3: BatchSize, PointsNumber, Dimension(4) Right Source

        K: K near points
    output:
        Concact Result: Batchsize, PointsNumber, Dimension + K*4
    """
    # idx: BatchSize, PointNumber, K,
    _, leftidx, _ = knn_points(pointsAnchor[:, :, :3], pointsLeft[:, :, :3], K=K, return_nn=True)
    _, rightidx, _ = knn_points(pointsAnchor[:, :, :3], pointsRight[:, :, :3], K=K, return_nn=True)

    nn_gather_feature_left = knn_gather(pointsLeft, leftidx)
    nn_gather_feature_right = knn_gather(pointsRight, rightidx)

    return nn_gather_feature_left, nn_gather_feature_right


def getAugResult(BatchData):
    """
    BatchData: BatchSize, PointNumber, Dimension(4)
    """
    BatchSize, PointNumber, Dimension = BatchData.shape
    paddings = torch.zeros(size=[1, PointNumber, Dimension]).to(BatchData.device)
    LeftData = torch.cat([paddings, BatchData[:BatchSize - 1]], dim=0)
    RightData = torch.cat([BatchData[1:], paddings], dim=0)

    # BatchSize,PointNumber,3,Dimension
    nn_gather_left, nn_gather_right = getKnnRes(BatchData, LeftData, RightData)
    BatchDataExpand = BatchData.unsqueeze(2).repeat(1, 1, 3, 1)
    BatchLeft = (BatchDataExpand - nn_gather_left).reshape(BatchSize, PointNumber, -1)
    result = torch.cat([BatchData, BatchLeft], dim=-1)

    return result


class Self_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads) -> None:
        super(Self_Attention, self).__init__()
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = hidden_size
        self.num_heads = num_heads
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def forward(self, X):
        """
        Input:
            X: batchsize, frame, input_size
        Output:
            key: batchsize, frame, hidden_size
            query: batchsize, frame, hidden_size
            value: batchsize, frame, hidden_size
        """
        key, query, value = self.key_layer(X), self.query_layer(X), self.value_layer(X)
        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attentionScores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attentionScores = attentionScores / math.sqrt(self.attention_head_size)
        attentionProbs = F.softmax(attentionScores, dim=-1)
        context = torch.matmul(attentionProbs, value_heads)  # batchsize, num_heads,frame,attention_head_size
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_size)

        return context

    def trans_to_multiple_heads(self, X):
        """
        X: batchsize, frame, hidden_size
        """
        new_size = X.size()[:-1] + (self.num_heads, self.attention_head_size)  # batchsize, frame, num_heads, attention_head_size
        X = X.view(new_size)
        return X.permute(0, 2, 1, 3)  # batchsize, num_heads,frame,attention_head_size


class LSTMMeanModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMMeanModel, self).__init__()
        self.BiLSTM = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, X):
        self.BiLSTM.flatten_parameters()
        outdata, (hidden, cells) = self.BiLSTM(X)
        hiddens = hidden.transpose(1, 0)
        hiddens = torch.sum(hiddens, dim=1)
        return hiddens


class LSTransBlock(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super(LSTransBlock, self).__init__()
        self.translayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                     batch_first=True)
        self.BiLSTM = nn.LSTM(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, X):
        """
        X : BatchSize, FrameNumber, Feature
        """
        self.BiLSTM.flatten_parameters()
        res = self.translayer(X)
        outdata, (hidden, cells) = self.BiLSTM(res)
        hidden = torch.sum(hidden, dim=0)
        return outdata, hidden


class MainModel(nn.Module):
    def __init__(self, input_channel, num_class, num_blocks):
        super(MainModel, self).__init__()
        self.ContentEncoder = PointFeature(input_channel=input_channel)
        self.LSTransBlock1 = LSTransBlock(d_model=256, n_heads=4)
        self.predictLayers = nn.Sequential(nn.Linear(128 * num_blocks, 64), nn.ReLU(inplace=True),
                                           nn.Linear(64, num_class))

    def forward(self, Points):
        """
        Points: BatchSize,45,180,4
        """
        PointsY = Points[:, :, :, 1]
        mindata, _ = torch.min(PointsY, dim=2)
        mindata, _ = torch.min(mindata, dim=1)
        Points[:, :, :, 1] = Points[:, :, :, 1] - mindata.reshape(-1, 1, 1)
        BatchRes = []
        for i in range(Points.shape[0]):
            AugPoints = getAugResult(Points[i])
            Feature = self.ContentEncoder(AugPoints)
            BatchRes.append(Feature)
        BatchRes = torch.stack(BatchRes, dim=0)

        outres1, hiddenres1 = self.LSTransBlock1(BatchRes)
        clres = self.predictLayers(hiddenres1)
        return clres, hiddenres1


class MoCo(nn.Module):
    def __init__(self, input_channel, num_class, base_encoder, dim=128, K=8192, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)  基本编码器输出的结果维度,128是LSTM输出的维度
        K: queue size: number of negative keys K字典队列的大小,默认为4096
        m: moco momentum of update 动量更新参数
        T: 维度参数
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(input_channel=input_channel, num_class=num_class, num_blocks=1)
        self.encoder_k = base_encoder(input_channel=input_channel, num_class=num_class, num_blocks=1)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("dataqueue", torch.randn(dim, K))
        self.register_buffer("labelqueue", torch.zeros(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keyslabel):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0

        self.dataqueue[:, ptr:ptr + batch_size] = keys.T
        self.labelqueue[ptr:ptr + batch_size] = keyslabel
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, q_label, is_train=True):
        """
        Input:
            im_q: a batch of query point clouds
            im_k: a batch of key point clouds
            q_label: im_q和im_k都具有相同的label
            is_train:
        """

        cls_q, q = self.encoder_q(im_q)
        if is_train:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                _, k = self.encoder_k(im_k)

            queRes = torch.einsum("nc,ck->nk", [q, self.dataqueue.clone().detach()])
            quelabel = self.labelqueue.clone().detach()  # shape K

            quelabel = torch.stack([quelabel for i in range(queRes.shape[0])], dim=0)

            predictlabel = torch.where(quelabel == q_label.reshape(-1, 1), 1, -1)
            predictlabel = predictlabel.float()

            self._dequeue_and_enqueue(k, q_label)

            return cls_q, q, queRes, predictlabel
        else:
            return cls_q


if __name__ == "__main__":
    input_channel = 16
    num_class = 23  # Modify this based on the number of classes in your dataset
    model = MoCo(input_channel, num_class, MainModel)

    # Example data
    data = torch.rand(size=[64, 45, 256, 4])
    data_k = data.clone()
    q_label = torch.randint(0, num_class, size=[64])

    # Run the model
    output = model(data, data_k, q_label, is_train=True)

    # Print output shapes
    print("Model")
    print("Output shapes:", [x.shape for x in output])