import torch
import torch.nn as nn
import torch.nn.functional as F

class PointFeature(nn.Module):
    def __init__(self, input_channel):
        super(PointFeature, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(2, 1)
        return x

def getAugResult(BatchData):
    device = BatchData.device
    BatchSize, PointNumber, Dimension = BatchData.shape
    paddings = torch.zeros(size=[1, PointNumber, Dimension], device=device)
    LeftData = torch.cat([paddings, BatchData[:BatchSize - 1]], dim=0)
    RightData = torch.cat([BatchData[1:], paddings], dim=0)

    BatchDataExpand = BatchData.unsqueeze(2).repeat(1, 1, 3, 1)
    BatchLeft = (BatchDataExpand - LeftData.unsqueeze(2)).reshape(BatchSize, PointNumber, -1)
    result = torch.cat([BatchData, BatchLeft], dim=-1)
    return result

class LSTransBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(LSTransBlock, self).__init__()
        self.translayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.BiLSTM = nn.LSTM(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, X):
        self.BiLSTM.flatten_parameters()
        res = self.translayer(X)
        outdata, (hidden, _) = self.BiLSTM(res)
        hidden = torch.sum(hidden, dim=0)
        return outdata, hidden

class MainModel(nn.Module):
    def __init__(self, input_channel, num_class, num_blocks):
        super(MainModel, self).__init__()
        self.ContentEncoder = PointFeature(input_channel=16)  
        self.LSTransBlock1 = LSTransBlock(d_model=256, n_heads=4)
        self.predictLayers = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True),
                                           nn.Linear(64, num_class))

    def forward(self, Points):
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
        # Reshape BatchRes to [batch_size, seq_len, d_model=256] to match transformer expectations
        BatchRes = BatchRes.reshape(BatchRes.size(0), BatchRes.size(1), -1)
        outres1, hiddenres1 = self.LSTransBlock1(BatchRes)
        clres = self.predictLayers(hiddenres1)
        return clres
    
if __name__ == "__main__":
    model = MainModel(input_channel=16, num_class=23, num_blocks=1)  # Model expects 16 input channels
    model.eval()
    dummy_input = torch.randn(1, 45, 16, 4).to('cuda')  # Adjust input shape according to your data
    model.to('cuda')
    torch.onnx.export(model, dummy_input, "main_model.onnx",
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)
    print("Main model has been converted to ONNX format and saved as main_model.onnx")
