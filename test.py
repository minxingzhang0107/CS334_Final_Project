import numpy
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
from coral_pytorch.dataset import proba_to_label

transform = nn.Sequential(transforms.Resize(56),
                          transforms.CenterCrop(56),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          )

dataset_train = PawpularityDataset(csv_file='y_train.csv',
                                   root_dir='petfinder-pawpularity-score/train/',
                                   transform=transform)

train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)

dataset_test = PawpularityDataset(csv_file='y_test.csv',
                                  root_dir='petfinder-pawpularity-score/train/',
                                  transform=transform)

test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=4)


# test
def compute_mae_and_rmse(model, data_loader, device):
    with torch.no_grad():
        mae, mse, acc, num_examples = 0., 0., 0., 0
        model.eval()

        for i, data in enumerate(data_loader):
            image = data['image'].to(device)
            score = data['score'].float().to(device)

            logits, probas = model(image)
            predicted_labels = proba_to_label(probas).float()

            num_examples += score.size(0)
            mae += torch.sum(torch.abs(predicted_labels - score)).item()
            mse += torch.sum((predicted_labels - score) ** 2).item()

        mae = mae / num_examples
        rmse = numpy.sqrt(mse / num_examples)
        return mae, rmse


# modify train method from train.py to add a return statement
def train(learning_rate, num_epochs, batch_size, NUM_CLASSES):
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

    return model


if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train(learning_rate=0.005, num_epochs=1, batch_size=64, NUM_CLASSES=101)
    train_mae, train_rmse = compute_mae_and_rmse(model, train_loader, DEVICE)
    test_mae, test_rmse = compute_mae_and_rmse(model, test_loader, DEVICE)
    print(f'Mean absolute error (train/test): {train_mae:.2f} | {test_mae:.2f}')
    print(f'Mean root squared error (train/test): {train_rmse:.2f} | {test_rmse:.2f}')
