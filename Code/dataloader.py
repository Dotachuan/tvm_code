#encoding=utf-8


import torch
from torch.utils.data import Dataset,DataLoader


class RadarDataSet(Dataset):
    def __init__(self,fileName) -> None:
        super().__init__()
        self.fileNameList = []
        with open(fileName,"r",encoding="utf-8") as f:
            self.fileNameList = f.readlines()

    def __getitem__(self, index):
        Segments = self.fileNameList[index*45:index*45+45]
        label = []
        data = []
        for file in Segments:
            data.append(torch.load(file.split("\t")[0]))
            label.append(file.split("\t")[1])
        data = torch.stack(data,dim=0)

        return data.float(),int(label[0])
    def __len__(self):
        return len(self.fileNameList) // 45
    


if __name__ == "__main__":
    #daset = RadarDataSet("../Records/Train316CFAR002_P15.txt")
    daset = RadarDataSet("../Records/Train316CFAR002.txt")
    dalo = DataLoader(daset,batch_size=256,num_workers=4,pin_memory=True)
    for idx, (traindata,trainlabel) in enumerate(dalo):
        traindata = traindata.cuda()
        trainlabel = trainlabel.cuda()
        print("Idx: ",idx, "traindata.shape: ",traindata.shape, " trainlable: ",trainlabel.shape)
   