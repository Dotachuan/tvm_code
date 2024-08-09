import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.lstm_trans_block = LSTransBlock(d_model=5, n_heads=1)

    def forward(self, x):
        x = self.fc(x)
        outdata, hidden = self.lstm_trans_block(x)
        return hidden

if __name__ == "__main__":
    model = TestModel()
    model.eval()
    dummy_input = torch.randn(1, 1, 10).to('cuda')
    model.to('cuda')
    torch.onnx.export(model, dummy_input, "test_model_with_trans_block.onnx",
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)
    print("Model with Transformer block has been converted to ONNX format and saved as test_model_with_trans_block.onnx")
