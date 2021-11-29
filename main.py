from train import train
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss
from coral_pytorch.layers import CoralLayer
from model import ConvNet # import from model.py
'''
# hyperparameters

learning_rate = 0.05
num_epochs = 10
batch_size = 128

# architecture
NUM_CLASSES = 10
'''
# train(learning_rate, num_epochs, batch_size, NUM_CLASSES)
def main():
	train(0.05, 10, 128, 10)