from model.conv_net import ConvNet
from model.dt import DecisionTreePawpularity
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PawpularityDataset
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss
import numpy
from coral_pytorch.dataset import proba_to_label
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt


class PawNet:
    def __init__(self, input_shape, lr=0.001, epochs=100, num_classes=101, batch_size=32, x_train_path='x_train.csv',
                 y_train_path='y_train.csv', x_test_path='x_test.csv', y_test_path='y_test.csv'):
        self.resolution = input_shape
        self.deep: ConvNet = ConvNet(input_shape, num_classes)
        self.shallow: DecisionTreePawpularity = DecisionTreePawpularity()
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.x_train = pd.read_csv(x_train_path)
        self.y_train = pd.read_csv(y_train_path)
        self.x_test = pd.read_csv(x_test_path)
        self.y_test = pd.read_csv(y_test_path)

    def train(self, evaluate):
        self.shallow.train(self.x_train, self.y_train)
        self.train_deep(evaluate=evaluate)

    def initialize_dataloader(self) -> (DataLoader, DataLoader):
        transform = nn.Sequential(transforms.Resize(self.resolution),
                                  transforms.CenterCrop(self.resolution),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  )

        dataset = PawpularityDataset(csv_file='y_train.csv',
                                     root_dir='petfinder-pawpularity-score/train/',
                                     transform=transform)
        test_dataset = PawpularityDataset(csv_file='y_test.csv',
                                          root_dir='petfinder-pawpularity-score/train/',
                                          transform=transform)

        # Initialize the dataloader
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=14)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=14)

        return train_loader, test_loader

    def train_deep(self, evaluate=False, train_loader=None, test_loader=None):
        if train_loader is None:
            train_loader, test_loader = self.initialize_dataloader()

        print('Training on', self.device)

        # optimizer
        optimizer = torch.optim.Adam(self.deep.parameters(), lr=self.lr, weight_decay=1e-5)
        # torch.manual_seed(random_seed)
        self.deep = self.deep.to(self.device)

        for epoch in range(self.epochs):
            self.deep.train()
            for batch_idx, data in enumerate(train_loader):
                # Convert class labels for CORAL
                optimizer.zero_grad()
                levels = levels_from_labelbatch(data['score'],
                                                num_classes=self.num_classes)

                image = data['image'].to(self.device)
                levels = levels.to(self.device)
                logits = self.deep(image)

                # CORAL loss
                loss = coral_loss(logits, levels)

                loss.backward()
                optimizer.step()

                # LOGGING
                if not batch_idx % 200:
                    print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                          % (epoch + 1, self.epochs, batch_idx,
                             len(train_loader), loss))

            # Test each epoch
            mae, rmse = self.predict(train_loader)
            print('Epoch: %03d/%03d | MAE: %.4f | RMSE: %.4f'
                  % (epoch + 1, self.epochs, mae, rmse))
            if evaluate:
                # Compute test mae and mse
                mae, rmse = self.predict(test_loader)
                print('Epoch: %03d/%03d | Test MAE: %.4f | Test RMSE: %.4f'
                      % (epoch + 1, self.epochs, mae, rmse))
        # torch.save(self.deep.state_dict(), 'deep_model.pt')

    def predict(self, data_loader):
        with torch.no_grad():
            mae, mse, acc, num_examples = 0., 0., 0., 0
            self.deep.eval()

            for i, data in enumerate(data_loader):
                image = data['image'].to(self.device)
                score = torch.flatten(data['score'].float().to(self.device))

                logits = self.deep(image)
                probas = torch.sigmoid(logits)
                predicted_labels = proba_to_label(probas).float()
                shallow_prediction = self.shallow.predict(data['metadata'])

                predicted_labels = (predicted_labels + torch.from_numpy(shallow_prediction).to(self.device))/2

                num_examples += score.size(0)
                mae += torch.sum(torch.abs(predicted_labels - score)).item()
                mse += torch.sum((predicted_labels - score) ** 2).item()

            mae = mae / num_examples
            rmse = numpy.sqrt(mse / num_examples)
            return mae, rmse

    def load_model(self, path: str):
        self.shallow.train()
        self.deep.load_state_dict(torch.load(path))

    def explain(self):
        batch_size = self.batch_size
        self.batch_size = 1
        cam = GradCAM(self.deep, target_layers=self.deep.features, use_cuda=True)

        # Load the test data
        test_loader = self.initialize_dataloader()[1]
        for i, test_data in enumerate(test_loader):
            if i > 10:
                break
            image = test_data['image']
            score = test_data['score'].int().item()

            # Get the CAM
            grayscale_cam = cam(input_tensor=image, target_category=score)
            grayscale_cam = grayscale_cam[0, :]
            input_image = numpy.transpose((image[0] / 255).numpy(), (1, 2, 0))
            visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
            plt.imshow(visualization)
            plt.show()

        self.batch_size = batch_size
