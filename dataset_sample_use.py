from dataset import PawpularityDataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model.sample_cnn import Net
import torch.optim as optim
import torch.nn as nn

# Initialize the dataset
transform = nn.Sequential(transforms.CenterCrop(256))

dataset = PawpularityDataset(csv_file='x_train.csv', root_dir='petfinder-pawpularity-score/train/', transform=transform)

# Initialize the dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Load sample CNN model
net = Net()

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'], data['score']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')