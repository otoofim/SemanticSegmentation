from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from PIL import *


class MapillaryLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, ver, transform_in = None, transform_ou = None, mode = 'tra'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ver = "v1.2" if ver else "v2.0"

        if mode == "tra":
            self.samples_path = os.path.join(root_dir, "training", "images")
            self.labels_path = os.path.join(root_dir, "training", self.ver, "labels")
        elif mode == "val":
            self.samples_path = os.path.join(root_dir, "validation", "images")
            self.labels_path = os.path.join(root_dir, "validation", self.ver, "labels")
        else:
            self.samples_path = os.path.join(root_dir, "testing", "images")
            self.labels_path = os.path.join(root_dir, "testing", self.ver, "labels")

        self.transform_in = transform_in
        self.transform_ou = transform_ou

    def __len__(self):
        return len(next(os.walk(self.samples_path))[2])
        #return 100

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = next(os.walk(self.samples_path))[2][idx].split('.')[0]

        base_img = Image.open(os.path.join(self.samples_path, img_name + ".jpg"))
        label_img = Image.open(os.path.join(self.labels_path, img_name + ".png")).convert('RGB')


        sample = {'image': base_img, 'label': label_img}


        if self.transform_in:
            sample['image'] = self.transform_in(sample['image'])
        if self.transform_ou:
            sample['label'] = self.transform_ou(sample['label'])

        return sample
