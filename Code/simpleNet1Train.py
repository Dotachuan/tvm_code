import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Simplified ContentEncoder class
class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=45, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

# Simplified LSTransBlock class
class LSTransBlock(nn.Module):
    def __init__(self):
        super(LSTransBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=64 * 25, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height, channels * width)  # Adjust the view operation to match LSTM input size
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last output of the LSTM
        return x


# Simplified GetAugResult class
class GetAugResult(nn.Module):
    def __init__(self):
        super(GetAugResult, self).__init__()

    def forward(self, x):
        noise = torch.randn_like(x) * 0.1  # Generate noise of the same size as x
        return x + noise

# Simplified MainModel class
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.getAugResult = GetAugResult()
        self.ContentEncoder = ContentEncoder()
        self.LSTransBlock = LSTransBlock()
        self.fc_final = nn.Linear(64, 10)  # Final classification layer

    def forward(self, x):
        x = self.getAugResult(x)
        x = self.ContentEncoder(x)
        x = self.LSTransBlock(x)
        x = self.fc_final(x)
        return x

# Simplified RadarDataSet class for loading the dataset
class RadarDataSet(Dataset):
    def __init__(self, fileName) -> None:
        super().__init__()
        self.fileNameList = []
        with open(fileName, "r", encoding="utf-8") as f:
            self.fileNameList = f.readlines()

    def __getitem__(self, index):
        Segments = self.fileNameList[index * 45: index * 45 + 45]
        label = []
        data = []
        for file in Segments:
            data.append(torch.load(file.split("\t")[0]))
            label.append(file.split("\t")[1])
        data = torch.stack(data, dim=0)
        return data.float(), int(label[0])

    def __len__(self):
        return len(self.fileNameList) // 45

# Accuracy functions
def accuracyFunction(outputvalue, targetValue):
    outputValue = F.log_softmax(outputvalue, dim=1)
    max_value, max_index = torch.max(outputValue, dim=1)
    acc = max_index == targetValue
    acc = torch.sum(acc) / acc.shape[0]
    return acc.item()

def accuracyCount(outputvalue, targetValue):
    outputValue = F.log_softmax(outputvalue, dim=1)
    max_value, max_index = torch.max(outputValue, dim=1)
    acc = max_index == targetValue
    return torch.sum(acc).item(), outputvalue.shape[0]

# Training parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 50  # Reduced number of epochs for simplicity
input_channel = 45
num_class = 10

# Paths to dataset (adjust these paths as needed)
train_data_path = "../Records/Train316CFAR005.txt"
test_data_path = "../Records/Test316CFAR005.txt"

# Create DataLoader
trainset = RadarDataSet(train_data_path)
testset = RadarDataSet(test_data_path)
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=batch_size, num_workers=4, pin_memory=True)

# Instantiate the model, loss function, and optimizer
model = MainModel().cuda()
lossCro = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

bestacc = 0.0

# Training loop
for epo in range(num_epochs):
    model.train()
    running_loss = 0.0
    for trainidx, (traindata, trainlabel) in enumerate(trainloader):
        traindata = traindata.cuda().float()
        trainlabel = trainlabel.cuda()
        optimizer.zero_grad()

        outputs = model(traindata)
        loss = lossCro(outputs, trainlabel)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if trainidx % 10 == 9:  # Print every 10 mini-batches
            print(f"[{epo + 1}, {trainidx + 1}] loss: {running_loss / 10:.4f}")
            running_loss = 0.0

    # Evaluation loop
    model.eval()
    acccount = 0
    allcount = 0
    with torch.no_grad():
        for testidx, (testdata, testlabel) in enumerate(testloader):
            testdata = testdata.cuda().float()
            testlabel = testlabel.cuda()
            testres = model(testdata)
            testloss = lossCro(testres, testlabel)
            accNum, batchNum = accuracyCount(testres, testlabel)
            acccount += accNum
            allcount += batchNum

            print(f"Epoch: {epo} testidx: {testidx} testloss: {testloss.item():.4f}")
        accres = acccount / allcount

        print(f"Accuracy: {accres:.4f}")

        if accres > bestacc:
            bestacc = accres
            torch.save(model, "../Result/CFAR_005PID_simplified.pth")

print("Training completed")
