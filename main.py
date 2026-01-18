from utils import *
from modules import FNO

# load data
xTrain, yTrain, inputShape = loadDataDarcy("darcy_train_16.pt")
print(xTrain.shape)

# create FNO model
model = FNO(inputShape)
print("model architecture", model)
tp = model(xTrain)

print(tp.shape)