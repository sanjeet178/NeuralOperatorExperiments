import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class SpectralConv2d(nn.Module):
    def __init__(self, projChannel, modesOne, modesTwo, coordOneDim, coordTwoDim, nBatch):
        super(SpectralConv2d, self).__init__()

        # Few input params
        self.modesOne = modesOne
        self.modesTwo = modesTwo
        self.projChannel = projChannel
        self.coordOneDim = coordOneDim
        self.coordTwoDim = coordTwoDim
        self.nBatch = nBatch

        # complex weight matrix
        self.complexWeight = nn.Parameter(torch.randn(self.projChannel, self.projChannel, self.modesOne, self.modesTwo, dtype=torch.cfloat))

    def forward(self, x):

        # apply fourier transform
        xFt = torch.fft.rfft2(x)

        # multiply weight with xFt. Truncation of higher modes is also take care here
        outFt = torch.zeros_like(x, dtype=torch.cfloat)
        weightMulxFt = torch.einsum("bixy,ioxy->boxy", xFt[:, :, :self.modesOne, :self.modesTwo], self.complexWeight)
        outFt[:, :, :self.modesOne, :self.modesTwo] = weightMulxFt

        # apply inverse fourier transform
        outIfft = torch.fft.irfft2(outFt, s=(self.coordOneDim, self.coordTwoDim)).real
    
        return outIfft

class OperatorBlock(nn.Module):
    def __init__(self, projChannel, modeOne, modeTwo,coordOneDim, coordTwoDim, nBatch):
        super(OperatorBlock, self).__init__()   

        # Few input params
        self.projChannel = projChannel
        self.modeOne = modeOne
        self.modeTwo = modeTwo

        # linear transformation
        self.w = nn.Conv2d(self.projChannel, self.projChannel, 1)
        self.spectralConv = SpectralConv2d(self.projChannel, self.modeOne, self.modeTwo,coordOneDim, coordTwoDim, nBatch)

    def forward(self, x):

        # linear conv and spectral conv
        outOne = self.w(x)
        outTwo = self.spectralConv(x)
        
        # output of both convolutions added
        logging.info(f"{outOne.shape}, {outTwo.shape}")
        output = F.gelu(outOne + outTwo)

        return output


class FNO(nn.Module):
    def __init__(self, inputShape, outputShape, modeOne, modeTwo):
        super(FNO, self).__init__()    

        # Few input params
        self.inputChannels = inputShape['channels']
        self.outputChannels = outputShape['channels']
        self.projChannel = 6
        self.modeOne = modeOne  
        self.modeTwo = modeTwo 

        # Lifting layer 
        self.lift = nn.Sequential(
            nn.Linear(self.inputChannels, self.projChannel),
            nn.GELU()
        )

        # Projection layer 
        self.projection = nn.Sequential(
            nn.Linear(self.projChannel, self.outputChannels),
            nn.GELU()
        )

        # Operator block
        self.fnoBlock = nn.ModuleList(
            [
                OperatorBlock(
                    self.projChannel, 
                    self.modeOne, 
                    self.modeTwo, 
                    inputShape['coordOneDim'], 
                    inputShape['coordTwoDim'], 
                    inputShape['nBatch']
                ) for i in range(1)
            ]
        )

    def forward(self, x):

        # lifting layer
        permutedX = torch.permute(x, (0,2,3,1))
        liftPermutedX = self.lift(permutedX)
        liftX = torch.permute(liftPermutedX, (0,3,1,2))

        # FNO blocks
        for i, l in enumerate(self.fnoBlock):
            fnoX = self.fnoBlock[i](liftX)

        # projection layer
        permutedfnoX = torch.permute(fnoX, (0,2,3,1))
        projPermutedX = self.projection(permutedfnoX)
        yComp = torch.permute(projPermutedX, (0,3,1,2))

        return yComp
    