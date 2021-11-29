import torch
from coral_pytorch.layers import CoralLayer
from torchvision import transforms
from torch.utils.data import DataLoader


class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        '''
        torch.nn.Conv2d(1, 3, (3, 3), (1, 1), 1), # L and W dimensions stay the same after conv layer
        torch.nn.MaxPool2d((2, 2), (2, 2)), # L and W dimensions divided by 2
        torch.nn.Conv2d(3, 6, (3, 3), (1, 1), 1),
        torch.nn.MaxPool2d((2, 2), (2, 2))) # 7x7x6
        # size is 7x7x6=294 after flattening
        '''

        self.features = torch.nn.Sequential(
            # input resolution: 56x56
            torch.nn.Conv2d(3, 6, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(6, 12, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(12, 24, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2))
        )
            # size: 7x7x24=1176
        
        # conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        # dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        
        # MaxPool2d: torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        ### Specify CORAL layer
        self.fc = CoralLayer(size_in=1176, num_classes=num_classes)
        ###--------------------------------------------------------------------###

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten

        ##### Use CORAL layer #####
        logits = self.fc(x)
        probas = torch.sigmoid(logits)
        ###--------------------------------------------------------------------###

        return logits, probas
