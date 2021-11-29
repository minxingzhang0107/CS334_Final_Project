from .conv_net import ConvNet
from .dt import DecisionTreePawpularity
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PawpularityDataset
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss


class PawNet:
    def __init__(self, input_shape, lr=0.001, epochs=20, num_classes=101, x_train_path='x_train.csv',
                 y_train_path='y_train.csv', x_test_path='x_test.csv', y_test_path='y_test.csv'):
        self.deep: ConvNet = ConvNet(input_shape)
        self.shallow: DecisionTreePawpularity = DecisionTreePawpularity()
        self.num_classes = num_classes
        self.epochs = epochs

        self.x_train = pd.read_csv(x_train_path)
        self.y_train = pd.read_csv(y_train_path)
        self.x_test = pd.read_csv(x_test_path)
        self.y_test = pd.read_csv(y_test_path)

    def train(self):
        self.shallow.train(self.x_train, self.y_train)
        self.train_deep()

    def initialize_dataloader(self) -> DataLoader:
        transform = nn.Sequential(transforms.Resize(56),
                                  transforms.CenterCrop(56),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  )

        dataset = PawpularityDataset(csv_file='y_train.csv',
                                     root_dir='petfinder-pawpularity-score/train/',
                                     transform=transform)

        # Initialize the dataloader
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

        return train_loader

    def train_deep(self):
        train_loader = self.initialize_dataloader()

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Training on', DEVICE)

        # torch.manual_seed(random_seed)
        model = ConvNet(num_classes=self.num_classes).to(DEVICE)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # torch.manual_seed(random_seed)
        model = model.train()

        for epoch in range(self.epochs):
            for batch_idx, data in enumerate(train_loader):
                # Convert class labels for CORAL
                levels = levels_from_labelbatch(data['score'],
                                                num_classes=self.num_classes)
                # --------------------------------------------------------------------###

                image = data['image'].to(DEVICE)
                levels = levels.to(DEVICE)
                logits, probas = model(image)

                # CORAL loss
                loss = coral_loss(logits, levels)
                # --------------------------------------------------------------------###

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # LOGGING
                if not batch_idx % 200:
                    print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                          % (epoch + 1, self.epochs, batch_idx,
                             len(train_loader), loss))

        self.model = model

    def predict(self):
        pass
