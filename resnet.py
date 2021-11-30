from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Block(torch.nn.Module):
  def __init__(self, n):
    super(Block, self).__init__()
    
    self.k = n*3
    self.p = n*6
    self.conv = torch.nn.Conv2d(self.k, self.p, (4, 4), (2, 2), 1)
    self.activation = torch.nn.ReLU()  
    self.conv2d_2 = torch.nn.Conv2d(self.k, self.p, (1, 1), (2, 2), 0) # 1x1 convolution

  def forward(self, x):
        y = self.conv(x)
        y = y + self.conv2d_2(x)
        return self.activation(y)
    

# model


from coral_pytorch.layers import CoralLayer


class ConvNet(torch.nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
      
        
        self.features = torch.nn.Sequential(
            Block(1),
            Block(2),
            Block(4)

        )

        


            # size after sequential layer: (resolution/8)x(resolution/8)x24
        
        # conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        # dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        
        # MaxPool2d: torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        ### Specify CORAL layer
        # self.fc = CoralLayer(size_in=(resolution/8)*(resolution/8)*24, num_classes=num_classes)
        ###--------------------------------------------------------------------###
        
        self.fc = CoralLayer(size_in=1176, num_classes=101) # returns error when the argument is "size_in=(56/8)*(56/8)*7"

    def forward(self, x):
        y = self.features(x)
        y = y.view(y.size(0), -1)  # flatten

        # use CORAL layer
        logits = self.fc(y)
        
        probas = torch.sigmoid(logits)
        

        return logits, probas
