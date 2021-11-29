import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss


for epoch in range(num_epochs):

    model = model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):

        ##### Convert class labels for CORAL
        levels = levels_from_labelbatch(class_labels, 
                                        num_classes=NUM_CLASSES)
        ###--------------------------------------------------------------------###

        features = features.to(DEVICE)
        levels = levels.to(DEVICE)
        logits, probas = model(features)

        #### CORAL loss 
        loss = coral_loss(logits, levels)
        ###--------------------------------------------------------------------###   


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        if not batch_idx % 200:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), loss))