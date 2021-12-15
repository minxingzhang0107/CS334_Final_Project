import numpy

from model.paw_net import PawNet
from sklearn.model_selection import KFold
from torch import nn
from dataset import PawpularityDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def kfold_test(model=None):
    resolution = 256
    batch_size = 32
    transform = nn.Sequential(transforms.Resize(resolution),
                              transforms.CenterCrop(resolution),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              )
    dataset = PawpularityDataset(csv_file='y_train.csv',
                                 root_dir='petfinder-pawpularity-score/train/',
                                 transform=transform)
    test_dataset = PawpularityDataset(csv_file='y_test.csv',
                                      root_dir='petfinder-pawpularity-score/train/',
                                      transform=transform)
    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    rmses = []
    dataset = numpy.array(dataset)
    for train_index, test_index in kf.split(dataset):
        train_dataset, val_dataset = dataset[train_index], dataset[test_index]

        # Initialize the dataloader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=14)

        # Extract x and y
        train_len = len(train_dataset)
        val_len = len(val_dataset)

        x_train, y_train = numpy.zeros((train_len, 12)), numpy.zeros((train_len,))
        x_val, y_val = numpy.zeros((val_len, 12)), numpy.zeros((val_len,))
        for index, sample in enumerate(train_dataset):
            x_train[index] = sample['metadata'].numpy()
            y_train[index] = sample['score'].cpu().item()

        for index, sample in enumerate(val_dataset):
            x_val[index] = sample['metadata'].numpy()
            y_val[index] = sample['score'].cpu().item()

        paw_net = PawNet(input_shape=256, epochs=12)
        if model is not None:
            paw_net.deep = model(resolution=256)

        paw_net.shallow.train(x_train, y_train)
        paw_net.train_deep(train_loader=train_loader, test_loader=val_loader, evaluate=True)
        mae, rmse = paw_net.predict(val_loader)
        rmses.append(rmse)
        print(f'RMSE: {rmse}')
    print(f'Average RMSE: {numpy.mean(rmses)}')
    print(f'Standard Deviation: {numpy.std(rmses)}')


if __name__ == '__main__':
    kfold_test()

    # paw_net.train(evaluate=True)
    # paw_net.load_model('deep_model.pt')
    # paw_net.deep.to(paw_net.device)
    # paw_net.explain()
    # paw_net.predict(paw_net.initialize_dataloader()[1])
