from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from PIL import *


class MapillaryLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, ver, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ver = "v1.2" if ver else "v2.0"

        self.samples_path = os.path.join(root_dir, "training", "images")
        self.labels_path = os.path.join(root_dir, "training", self.ver, "labels")

        self.transform = transform

    def __len__(self):
        return len(next(os.walk(self.samples_path))[2])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = next(os.walk(self.samples_path))[2][idx].split('.')[0]

        base_img = Image.open(os.path.join(self.samples_path, img_name+".jpg"))
        label_img = Image.open(os.path.join(self.labels_path, img_name+".png"))


        sample = {'image': base_img, 'label': label_img}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            #sample['label'] = self.transform(sample['label'])

        return sample
