import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np


### residual_blocks=True # ResNet style basic blocks
### block_reps=2 # Deeper network


class SubmanifoldUNet(nn.Module):
    def __init__(self, in_channels=4, f_maps=32, residual_blocks=False, block_reps=1, dimension=3, full_scale=4096):
        nn.Module.__init__(self)
        
        self.inp_layer = scn.InputLayer(dimension, spatial_size=full_scale, mode=4)
        ### Takes a tuple (coords, features, batch_size [optional])
        ### coords: N x (dimension + 1)  (first d columns are coordinates, last column is batch index)
        
        self.subm_conv = scn.SubmanifoldConvolution(dimension, in_channels, f_maps, 3, False)
        ### dimension, nIn, nOut, filter_size, bias
        
        self.unet = scn.UNet(dimension, block_reps, [f_maps, 2*f_maps, 3*f_maps, 4*f_maps,
                                                     5*f_maps, 6*f_maps, 7*f_maps], residual_blocks)
        self.bn_relu = scn.BatchNormReLU(f_maps)
        self.output_layer = scn.OutputLayer(dimension)
        
    def forward(self,x):
        x = self.inp_layer(x)
        x = self.subm_conv(x)
        x = self.unet(x)
        x = self.bn_relu(x)
        x = self.output_layer(x)
        return x