#encoding=utf-8

import torch
import numpy as np
from pytorch3d.ops import knn_points,knn_gather


import torch
import numpy as np
from pytorch3d.ops import knn_points, knn_gather

def getKnnRes(pointsAnchor, pointsLeft, pointsRight, K=3):
    _, leftidx, _ = knn_points(pointsAnchor[:, :, :3], pointsLeft[:, :, :3], K=K, return_nn=True)
    _, rightidx, _ = knn_points(pointsAnchor[:, :, :3], pointsRight[:, :, :3], K=K, return_nn=True)

    nn_gather_feature_left = knn_gather(pointsLeft, leftidx)
    nn_gather_feature_right = knn_gather(pointsRight, rightidx)

    return nn_gather_feature_left, nn_gather_feature_right

def getAugResult(BatchData):
    '''
    BatchData: BatchSize, PointNumber, Dimension(4)
    '''
    device = BatchData.device
    BatchSize, PointNumber, Dimension = BatchData.shape
    paddings = torch.zeros(size=[1, PointNumber, Dimension], device=device)
    LeftData = torch.cat([paddings, BatchData[:BatchSize - 1]], dim=0)
    RightData = torch.cat([BatchData[1:], paddings], dim=0)

    nn_gather_left, nn_gather_right = getKnnRes(BatchData, LeftData, RightData)
    BatchDataExpand = BatchData.unsqueeze(2).repeat(1, 1, 3, 1)
    BatchLeft = (BatchDataExpand - nn_gather_left).reshape(BatchSize, PointNumber, -1)
    result = torch.cat([BatchData, BatchLeft], dim=-1)

    return result

# New getAugResult
# def getAugResult(BatchData):
#     '''
#     BatchData: BatchSize,PointNumber,Dimension(4)
#     '''
#     device = BatchData.device
#     BatchSize, PointNumber, Dimension = BatchData.shape
#     paddings = torch.zeros(size=[1, PointNumber, Dimension], device=device)
#     LeftData = torch.cat([paddings, BatchData[:BatchSize - 1]], dim=0)
#     RightData = torch.cat([BatchData[1:], paddings], dim=0)

#     # Get KNN results
#     nn_gather_left, nn_gather_right = getKnnRes(BatchData, LeftData, RightData)

#     # --- Batch-wise Processing ---
#     BatchLeft = []
#     BatchRight = []
#     for i in range(BatchSize):
#         start_idx = i * PointNumber
#         end_idx = (i + 1) * PointNumber
        
#         # Slice for the current batch element
#         BatchDataSlice = BatchData[i].unsqueeze(0) # Add a batch dimension for slicing compatibility
#         BatchDataExpand = BatchDataSlice.unsqueeze(2).repeat(1, 1, 3, 1) 
#         nn_gather_left_slice = nn_gather_left[start_idx:end_idx].reshape(1, PointNumber, 3, Dimension)
#         nn_gather_right_slice = nn_gather_right[start_idx:end_idx].reshape(1, PointNumber, 3, Dimension)

#         # Calculate BatchLeft and BatchRight for the current batch element
#         BatchLeft.append((BatchDataExpand - nn_gather_left_slice).reshape(1, PointNumber, -1))
#         BatchRight.append((BatchDataExpand - nn_gather_right_slice).reshape(1, PointNumber, -1))

#     # Stack the results along the batch dimension
#     BatchLeft = torch.cat(BatchLeft, dim=0)
#     BatchRight = torch.cat(BatchRight, dim=0)
#     # --- End of Batch-wise Processing ---

#     result = torch.cat([BatchData, BatchLeft], dim=-1)

#     return result

if __name__ == "__main__":
    data = torch.rand(size=[128, 45, 180, 4])
    dataY = data[:, :, :, 1]
    mindata, minIndex = torch.min(dataY, dim=2)
    mindata, _ = torch.min(mindata, dim=1)
    data[:, :, :, 1] = data[:, :, :, 1] - mindata.reshape(-1, 1, 1)
    print(data.shape)




