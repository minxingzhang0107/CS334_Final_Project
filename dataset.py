from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class PawpularityDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. (should be y_train.csv)
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pawpularity_frame = pd.read_csv(csv_file)
        self.metadata = pd.read_csv(csv_file.replace("y", "x"))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pawpularity_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pawpularity_frame.iloc[idx, 0] + ".jpg")
        image = io.imread(img_name)
        score = np.array([self.pawpularity_frame.iloc[idx, 1]])
        metadata: pd.Series = self.metadata.iloc[idx, 1:]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        sample = {'image': torch.from_numpy(image),
                  'score': torch.LongTensor(score),
                  'metadata': torch.from_numpy(metadata.to_numpy().astype(np.int8))}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


if __name__ == "__main__":
    face_dataset = PawpularityDataset(csv_file='x_train.csv', root_dir='petfinder-pawpularity-score/train/')

    fig = plt.figure()

    for i in range(4):
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['score'])
        plt.imshow(sample['image'])
        plt.show()

