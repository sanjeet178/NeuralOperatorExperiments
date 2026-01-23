import torch
import os
import logging

def extractXDataShape(xCoordLess):

    inputShape = {}           
    inputShape['nBatch'], inputShape['channels'],inputShape['coordOneDim'], inputShape['coordTwoDim']  = xCoordLess.shape

    return inputShape

def loadDataDarcy(fileName:str):

    # ---------- filename handling ----------
    base_name = os.path.splitext(os.path.basename(fileName))[0]
    x_path = f"data/x_{base_name}.pt"
    y_path = f"data/y_{base_name}.pt"

    # ---------- if files already exist, just load ----------
    if os.path.exists(x_path) and os.path.exists(y_path):
        logging.info(f"Loading cached tensors:\n{x_path}\n{y_path}")
        x = torch.load(x_path, map_location="cpu")
        y = torch.load(y_path, map_location="cpu")

        # input shape and output shape
        inputShape = {}   
        outputShape = {}            
        inputShape['nBatch'], inputShape['channels'], inputShape['coordOneDim'], inputShape['coordTwoDim']  = x.shape
        outputShape['nBatch'], outputShape['channels'], outputShape['coordOneDim'], outputShape['coordTwoDim']  = y.shape

    else:

        # import data
        data = torch.load(os.path.join("data", fileName), map_location="cpu")
        xCoordLess = (data['x']*9+3).float().unsqueeze(1)
        y = data['y'].float().unsqueeze(1)

        # input shape and output shape
        inputShape = {}   
        inputShape['nBatch'], inputShape['channels'], inputShape['coordOneDim'], inputShape['coordTwoDim']  = xCoordLess.shape
       
        # Add coordinate channels to xCoordLess
        coordOne = torch.linspace(0, 1, steps=inputShape['coordOneDim'])
        coordTwo = torch.linspace(0, 1, steps=inputShape['coordTwoDim'])
        gridCoordOne, gridCoordTwo = torch.meshgrid(coordOne, coordTwo, indexing='ij')
        channelCoordOne = gridCoordOne.unsqueeze(0).repeat(inputShape['nBatch'], 1, 1, 1)
        channelCoordTwo = gridCoordTwo.unsqueeze(0).repeat(inputShape['nBatch'], 1, 1, 1)
        x = torch.cat([xCoordLess, channelCoordOne, channelCoordTwo], dim=1)

        # input shape and output shape  
        outputShape = {}            
        inputShape['nBatch'], inputShape['channels'], inputShape['coordOneDim'], inputShape['coordTwoDim']  = x.shape
        outputShape['nBatch'], outputShape['channels'], outputShape['coordOneDim'], outputShape['coordTwoDim']  = y.shape

        # Save tensors
        os.makedirs("data", exist_ok=True)
        torch.save(x, x_path)
        torch.save(y, y_path)
        
    return x, y, inputShape, outputShape


