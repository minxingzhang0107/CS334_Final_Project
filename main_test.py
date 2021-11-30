import torch
from train import train
from test import compute_mae_and_rmse
from dataset import PawpularityDataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


transform = nn.Sequential(transforms.Resize(56),
                          transforms.CenterCrop(56),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          )

dataset_train = PawpularityDataset(csv_file='y_train.csv',
                                   root_dir='petfinder-pawpularity-score/train/',
                                   transform=transform)

train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)

dataset_test = PawpularityDataset(csv_file='y_test.csv',
                                  root_dir='petfinder-pawpularity-score/train/',
                                  transform=transform)

test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=4)


if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train(train_loader, learning_rate=0.005, num_epochs=10, batch_size=64, NUM_CLASSES=101)
    train_mae, train_rmse = compute_mae_and_rmse(model, train_loader, DEVICE)
    test_mae, test_rmse = compute_mae_and_rmse(model, test_loader, DEVICE)
    print(f'Mean absolute error (train/test): {train_mae:.2f} | {test_mae:.2f}')
    print(f'Mean root squared error (train/test): {train_rmse:.2f} | {test_rmse:.2f}')