"""
@author: Trinh Khai Truong 
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from constants import CLASSES


class MyDataset(Dataset):
    def __init__(self, root, isTrain, ratio, max_samples_per_class=10000):
        self.root = root
        self.images = []
        self.labels = []
        self.max_samples_per_class = max_samples_per_class
        for i, cls in enumerate(CLASSES):
            images = np.load("{}/full_numpy_bitmap_{}.npy".format(self.root, cls))
            images = np.reshape(images, (-1, 28, 28))
            images = images[:max_samples_per_class]
            if isTrain:
                image_len = int(max_samples_per_class * ratio)
                images = images[:image_len]
            else:
                image_len = int(max_samples_per_class * (1 - ratio))
                images = images[-image_len:]
            self.images.append(images)
            self.labels.append(np.full((image_len, 1), i))
        
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)     

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = image.astype(np.float32) / 255.0
        image = np.reshape(image, (1, 28, 28))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label).long()
        return image, label
    

if __name__ == "__main__":
    root = "dataset"
    isTrain = True
    ratio = 0.8

    dataset = MyDataset(root, isTrain, ratio)
    sample = dataset[99000]
    image, label = sample

    print(image.shape)
    print(label.item())

