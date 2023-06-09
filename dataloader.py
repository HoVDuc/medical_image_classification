from torch.utils.data import Dataset
import torch
from torchvision import transforms 
from PIL import Image
import numpy as np


class MedicalData(Dataset):

    def __init__(self, path, df, classes, device, transform=None):
        self.df = df
        classes = classes['class'].to_list()
        self.convert_labels(classes)
        self.device = device
        self.path = path
        self.transform = transform

    def convert_labels(self, classes):
        self.class2index = {label: idx for idx, label in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]['file']
        folder = self.df.iloc[index]['label']
        label = torch.tensor(self.class2index[folder], device=self.device)
        file_path = f"{self.path}{folder}/{filename}"
        image = Image.open(file_path)
        image = np.array(image)
        to_tensor = transforms.Compose([
            transforms.ToTensor(), 
        ])
        if self.transform:
            image = self.transform(image=image)['image']
            image = to_tensor(image)
            
        image = image.to(self.device)
        label = label.to(self.device)
        return image, label
