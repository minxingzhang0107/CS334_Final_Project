import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss
from coral_pytorch.layers import CoralLayer
from model.conv_net import ConvNet  # import from conv_net.py
from dataset import PawpularityDataset
from torch import nn
from test import compute_mae_and_rmse

'''
# hyperparameters

learning_rate = 0.05
num_epochs = 10
batch_size = 128

# architecture
NUM_CLASSES = 10
'''


def train(train_loader, learning_rate, num_epochs, batch_size, NUM_CLASSES):
    random_seed = 1

    # device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on', DEVICE)

    torch.manual_seed(random_seed)
    model = ConvNet(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.manual_seed(random_seed)

    model = model.train()

    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(train_loader):
            ##### Convert class labels for CORAL
            levels = levels_from_labelbatch(data['score'],
                                            num_classes=NUM_CLASSES)
            ###--------------------------------------------------------------------###

            image = data['image'].to(DEVICE)
            levels = levels.to(DEVICE)
            logits, probas = model(image)

            #### CORAL loss
            loss = coral_loss(logits, levels)
            ###--------------------------------------------------------------------###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING
            if not batch_idx % 200:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss))

        # Test each epoch
        mae, rmse = compute_mae_and_rmse(model, train_loader, DEVICE)
        print('Epoch: %03d/%03d | MAE: %.4f | RMSE: %.4f'
              % (epoch + 1, num_epochs, mae, rmse))

    return model



if __name__ == '__main__':
    transform = nn.Sequential(transforms.Resize(56),
                              transforms.CenterCrop(56),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              )

    dataset = PawpularityDataset(csv_file='y_train.csv',
                                 root_dir='petfinder-pawpularity-score/train/',
                                 transform=transform)

    # Initialize the dataloader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    train(train_loader, learning_rate=0.005, num_epochs=10, batch_size=64, NUM_CLASSES=101)
