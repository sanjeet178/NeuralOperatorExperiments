from utils import *
from modules import FNO
import logging
import os

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
model.train()
totalTrainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters in the model: {totalTrainableParams}")