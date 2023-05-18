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

def undersampling(df):
    df_list = []
    for label in set(df['label'].values):
        sampling = df[df['label'] == label].head(df['label'].value_counts()['AIS']).reset_index(drop=True)
        df_list.append(sampling)

    df_sampling = pd.concat(df_list, axis=0).reset_index(drop=True)
    return df_sampling


def split_data(sampling=False):
    data = load_data()
    df = pd.DataFrame(data)
    
    if sampling:
        df = undersampling(df)
        
    df_sample = df.sample(frac=1).reset_index(drop=True)

    n_sample = len(df)
    n_train = int(n_sample * 0.7)
    n_valid = int(n_sample * 0.2)
    n_test = n_sample - (n_train + n_valid)

    df_train = df_sample[:n_train]
    df_valid = df_sample[n_train:n_train+n_valid]
    df_test = df_sample[n_train+n_valid:]

    return df_train, df_valid, df_test


def check_gpu():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    df_train, df_valid, df_test = split_data(True)
    device = check_gpu()

    transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.RandomRotation(30),
        # transforms.GaussianBlur(3),
        # transforms.RandomAffine((1, 30)),
        # transforms.RandomHorizontalFlip(0.3),
        # transforms.CenterCrop((224, 224)) if random.random() <= 0.3 else ,
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = MedicalData(PATH, df_train, device, transform)
    valid_data = MedicalData(PATH, df_valid, device, transform)
    test_data = MedicalData(PATH, df_test, device, transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=64,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=64)

    model = timm.create_model(
        'hf-hub:timm/eca_nfnet_l0', pretrained=True, num_classes=6)

    trainer = Trainer(model, train_loader, valid_loader, test_loader, 10, 1e-4, device, 3)
    trainer.train()


if __name__ == "__main__":
    main()
