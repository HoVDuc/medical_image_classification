from trainer import Trainer
from dataloader import MedicalData
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch
from PIL import Image
import pandas as pd
import numpy as np
import random
import pathlib
import glob
import os
import timm
import sys
sys.path.append('./pytorch_model/')


PATH = '../../../adminpc/workspace/Sharing_Data/Data_classification/'


def load_data():
    folders = glob.glob(PATH + "/*/")

    data = {
        'file': [],
        'label': []
    }
    for folder in folders:
        files = [os.path.basename(file) for file in glob.glob(folder + '/*')]
        for file in files:
            data['file'].append(file)
            data['label'].append(pathlib.PurePath(folder).name)

    return data


def split_data():
    data = load_data()
    df = pd.DataFrame(data)
    df_sample = df.sample(frac=1).reset_index(drop=True)

    n_sample = len(df)
    n_train = int(n_sample * 0.8)

    df_train = df_sample[:n_train]
    df_valid = df_sample[n_train:]

    return df_train, df_valid


def check_gpu():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    df_train, df_valid = split_data()
    device = check_gpu()

    transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.RandomRotation(30),
        transforms.GaussianBlur(3),
        # transforms.RandomAffine((1, 30)),
        # transforms.RandomHorizontalFlip(0.3),
        # transforms.CenterCrop((224, 224)) if random.random() <= 0.3 else ,
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = MedicalData(PATH, df_train, device, transform)
    valid_data = MedicalData(PATH, df_valid, device, transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=64,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=64)

    model = timm.create_model(
        'hf-hub:timm/eca_nfnet_l0', pretrained=True, num_classes=6)

    trainer = Trainer(model, train_loader, valid_loader, 10, 1e-4, device)
    trainer.train()


if __name__ == "__main__":
    main()
