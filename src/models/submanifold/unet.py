import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np


# m = 16 # 16 or 32
# residual_blocks=False #True or False
# block_reps = 1 #Conv block repetition factor: 1 or 2

# m=32 # Wider network
# residual_blocks=True # ResNet style basic blocks
# block_reps=2 # Deeper network

# dimension=3
# full_scale=4096 #Input field size


class SubmanifoldUNet(nn.Module):
    def __init__(self, m=32, residual_blocks=False, block_reps=1, dimension=3, full_scale=4096, in_channels=1):
        nn.Module.__init__(self)
        
        self.inp_layer = scn.InputLayer(dimension, spatial_size=full_scale, mode=4)
        ### Takes a tuple (coords, features, batch_size [optional])
        ### coords: N x (dimension + 1)  (first d columns are coordinates, last column is batch index)
        
        self.subm_conv = scn.SubmanifoldConvolution(dimension, in_channels, m, 3, False)
        ### dimension, nIn, nOut, filter_size, bias
        
        self.unet = scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)
        self.bn_relu = scn.BatchNormReLU(m)
        self.output_layer = scn.OutputLayer(dimension)
        
#         self.sparseModel = scn.Sequential().add(
#            scn.InputLayer(dimension, full_scale, mode=4)).add(
#            scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
#                scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
#            scn.BatchNormReLU(m)).add(
#            scn.OutputLayer(dimension))
#         self.linear = nn.Linear(m, 20)
        
    def forward(self,x):
        x = self.inp_layer(x)
#         print('\n#####################')
#         print(x.features.size(1)) # 1
#         print(self.subm_conv.nIn) # 1
#         print('#####################\n')
        x = self.subm_conv(x)
        x = self.unet(x)
        x = self.bn_relu(x)
        x = self.output_layer(x)
        
#         x = self.sparseModel(x)
#         x = self.linear(x)
        return x