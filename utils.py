import torch
import os

def extractXDataShape(xTrainCoordLess):

    inputShape = {}           
    inputShape['nBatch'], inputShape['inputChannels'],inputShape['coordOneDim'], inputShape['coordTwoDim']  = xTrainCoordLess.shape

    return inputShape

def loadDataDarcy(fileName:str):

    # ---------- filename handling ----------
    base_name = os.path.splitext(os.path.basename(fileName))[0]
    x_path = f"data/xTrain_{base_name}.pt"
    y_path = f"data/yTrain_{base_name}.pt"

    # ---------- if files already exist, just load ----------
    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading cached tensors:\n{x_path}\n{y_path}")
        xTrain = torch.load(x_path, map_location="cpu")
        yTrain = torch.load(y_path, map_location="cpu")

        # input shape
        inputShape = {}           
        inputShape['nBatch'], inputShape['inputChannels'],inputShape['coordOneDim'], inputShape['coordTwoDim']  = xTrain.shape
    else:
        # import training data
        data = torch.load("data/darcy_train_16.pt", map_location="cpu")
        xTrainCoordLess = (data['x']*9+3).float().unsqueeze(1)
        yTrain = data['y'].float().unsqueeze(1)

        # input shape
        inputShape = {}           
        inputShape['nBatch'], inputShape['inputChannels'],inputShape['coordOneDim'], inputShape['coordTwoDim']  = xTrainCoordLess.shape

        # Add coordinate channels to xTrainCoordLess
        coordOne = torch.linspace(0, 1, steps=inputShape['coordOneDim'])
        coordTwo = torch.linspace(0, 1, steps=inputShape['coordTwoDim'])
        gridCoordOne, gridCoordTwo = torch.meshgrid(coordOne, coordTwo, indexing='ij')
        channelCoordOne = gridCoordOne.unsqueeze(0).repeat(inputShape['nBatch'], 1, 1, 1)
        channelCoordTwo = gridCoordTwo.unsqueeze(0).repeat(inputShape['nBatch'], 1, 1, 1)
        xTrain = torch.cat([xTrainCoordLess, channelCoordOne, channelCoordTwo], dim=1)

        # Save tensors
        os.makedirs("data", exist_ok=True)
        torch.save(xTrain, x_path)
        torch.save(yTrain, y_path)
        
    return xTrain, yTrain, inputShape


