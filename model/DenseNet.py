import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, utils
from coral_pytorch.layers import CoralLayer

# input size 3136
class DenseNet(torch.nn.Module):

    def __init__(self):
        super(DenseNet, self).__init__()
      
        
        self.features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(9408, 2000), # Note: in_features is 9408 because the images has 3 channels
            torch.nn.Linear(2000, 1400),
            torch.nn.Linear(1400, 800)
        )

        self.fc = CoralLayer(size_in=800, num_classes=101) # returns error when the argument is "size_in=(56/8)*(56/8)*7"

    def forward(self, x):
        y = self.features(x)

        # use CORAL layer
        logits = self.fc(y)

        return logits
