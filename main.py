from utils import *
from modules import FNO

# load data
xTrain, yTrain, inputShape, outputShape = loadDataDarcy("darcy_train_16.pt")
print(xTrain.shape, yTrain.shape)

# create FNO model
model = FNO(inputShape, outputShape)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters in the model:", total_trainable_params)
print("model architecture", model)
yComp = model(xTrain)
# model.train()

print(yComp.shape)