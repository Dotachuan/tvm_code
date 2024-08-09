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
    # def forward(self, x):
    #     x = x.transpose(2, 1)
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     x = x.transpose(2, 1)
    #     return x

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

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(10, 5)
        self.lstm_trans_block = LSTransBlock(d_model=256, n_heads=1)
        self.ContentEncoder = PointFeature(input_channel=20) 
    def forward(self, x):
        x = self.fc(x)
        AugPoints = getAugResult(x)
        Feature = self.ContentEncoder(AugPoints)
        outdata, hidden = self.lstm_trans_block(Feature)
        return hidden

if __name__ == "__main__":
    model = TestModel()
    model.eval()
    dummy_input = torch.randn(1, 45, 10).to('cuda')  
    model.to('cuda')
    torch.onnx.export(model, dummy_input, "test_model_with_encoder.onnx",
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)
    print("Model with Encoder and Transformer block has been converted to ONNX format and saved as test_model_with_encoder.onnx")
