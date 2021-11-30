import numpy
import torch
from coral_pytorch.dataset import proba_to_label


# test
def compute_mae_and_rmse(model, data_loader, device):
    with torch.no_grad():
        mae, mse, acc, num_examples = 0., 0., 0., 0
        model.eval()

        for i, data in enumerate(data_loader):
            image = data['image'].to(device)
            score = torch.flatten(data['score'].float().to(device))

            logits, probas = model(image)
            predicted_labels = proba_to_label(probas).float()

            num_examples += score.size(0)
            mae += torch.sum(torch.abs(predicted_labels - score)).item()
            mse += torch.sum((predicted_labels - score) ** 2).item()

        mae = mae / num_examples
        rmse = numpy.sqrt(mse / num_examples)
        return mae, rmse
