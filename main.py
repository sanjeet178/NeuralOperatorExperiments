from utils import *
from modules import FNO
import logging
import os
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from postprocessing import plotAndSaveScatter

# Hyperparameters setup
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learningRate", type=float, default=0.001)
parser.add_argument("--modelTrain", type=bool, default=False)
parser.add_argument("--modelInference", type=bool, default=True)
parser.add_argument("--comparsionTestSize", type=int, default=5)
args = parser.parse_args()

# logs setup
logFile = "FNO.log"
logging.basicConfig(
    filename=logFile,
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get current working directory
currentDir = os.getcwd()

# load data
xTrain, yTrain, inputShape, outputShape = loadDataDarcy("darcy_train_16.pt")
logging.info(f"xTrain.shape, yTrain.shape: {xTrain.shape}, {yTrain.shape}")

# create FNO model
model = FNO(inputShape, outputShape)
logging.info(f"Model architecture: {model}")
totalTrainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters in the model: {totalTrainableParams}")

# Train data
if args.modelTrain==True:
    
    logging.info("Training has begun !!!!")

    # Training details setup
    model.train()
    optimiser = torch.optim.Rprop(model.parameters(), lr = args.learningRate)
    mse = nn.MSELoss()
    lossArray = np.zeros((args.epochs, 2))

    # Important directories
    saveDir = "./modelParams"
    os.makedirs(saveDir, exist_ok=True)

    # training loop
    for epoch in range(args.epochs):

        yComp = model(xTrain)
        loss = mse(yComp, yTrain)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        lossArray[epoch] = [epoch, loss.detach().numpy()]

        if epoch%10==0:
            logging.info(f"epoch = {epoch}, loss = {loss.detach().numpy()}")
            torch.save(model.state_dict(), "./modelParams/FNO_1Block_2D.pth")
            np.save(os.path.join(saveDir, "loss_history.npy"), lossArray)

# Inference
if args.modelInference==True:

    logging.info("Inference has begun !!!!")

    # Important directories
    saveDir = "./results"
    modelPath = "./modelParams/FNO_1Block_2D.pth"
    lossPath = "./modelParams/loss_history.npy"
    testDataName = "darcy_test_32.pt"
    baseTestDataName = os.path.splitext(os.path.basename(testDataName))[0]
    os.makedirs(saveDir, exist_ok=True)

    # load model and data
    modelPath = "./modelParams/FNO_1Block_2D.pth"
    if not os.path.isfile(modelPath):
        logging.error("Model checkpoint not found at path: %s", modelPath)
        raise FileNotFoundError(modelPath)
    model.load_state_dict(torch.load(modelPath, map_location="cpu"))
    model.eval()
    xTest, yTest, inputShape, outputShape = loadDataDarcy(testDataName)
    indices = torch.randperm(inputShape["nBatch"])[:args.comparsionTestSize].detach().numpy()
    selectedXTest = xTest[torch.tensor(indices)]
    selectedYTest = yTest[torch.tensor(indices)]
    xCoords, yCoords = selectedXTest[0,1], selectedXTest[0,2]
    selectedYComp = model(selectedXTest)

    # Scatter plots
    plotAndSaveScatter(selectedYComp, selectedYTest, saveDir, baseTestDataName)

    # Load loss history
    loss_history = np.load("./modelParams/loss_history.npy")
    epochs = loss_history[:, 0]
    losses = loss_history[:, 1]

    # Plot
    plt.figure()
    plt.scatter(epochs, losses, marker='o')
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Save figure
    plt.savefig(os.path.join(saveDir, "training_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()  

    logging.info("Inference has ended !!!!")