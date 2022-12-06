from PIL import Image
import os
import torch
from torchvision import transforms
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type, images, labels):
        self._dataset_type = dataset_type
        self._images = images
        self._labels = labels
        self._img_transforms = self.image_transforms()

    def __len__(self):
        return len(self._images)
    
    def image_transforms(self):
        if self._dataset_type == 'train':
            img_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])

        else:
            img_transforms = transforms.Compose([
                transforms.ToTensor()
            ])

        return img_transforms

    def __getitem__(self, idx):
        image_path = self._images[idx]
        temp_sample = self._img_transforms(Image.open(image_path).convert("RGB"))
        temp_label = torch.as_tensor(self._labels[idx])
        
        return image_path, temp_sample, temp_label