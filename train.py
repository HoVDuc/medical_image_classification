from torchvision import transforms
from dataloader import MedicalData
from trainer import Trainer
import timm
import torchvision
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import argparse
from PIL import Image
import glob
import pathlib
import random
import os
import sys
from sklearn.model_selection import KFold

sys.path.append('./pytorch_model/')


def load_data(PATH):
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
        sampling = df[df['label'] == label].head(
            df['label'].value_counts()['AIS']).reset_index(drop=True)
        df_list.append(sampling)

    df_sampling = pd.concat(df_list, axis=0).reset_index(drop=True)
    return df_sampling


def split_data(PATH, sampling=False):
    data = load_data(PATH)
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


def kfold_split(PATH):
    data = load_data(PATH)
    df = pd.DataFrame(data)
    df_sample = df.sample(frac=1).reset_index(drop=True)
    kfold = KFold()
    return df_sample, kfold


def check_gpu():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser(description='Classification training')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--path_data', type=str,
                        default='../Data_classification/')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--loss_function', type=str, default='ce')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./save/')
    parser.add_argument('--save_name', type=str, default='model.pt')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--print_every', type=int, default=5)
    parser.add_argument('--use_kfold', action='store_true')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    PATH = args['path_data']

    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])

    device = check_gpu()

    model = timm.create_model(
        args['model_name'], pretrained=True, num_classes=6)

    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.RandomRotation(30),
        transforms.GaussianBlur(3),
        # transforms.RandomAffine((1, 30)),
        transforms.RandomHorizontalFlip(0.3),
        # transforms.CenterCrop((224, 224)) if random.random() <= 0.3 else ,
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if args['use_kfold']:
        df, kfold = kfold_split(PATH)
        
        trainer = Trainer(model=model,
                      loss=args['loss_function'],
                      epochs=args['num_epochs'],
                      max_lr=args['lr'],
                      device=device,
                      num_samples=len(df),
                      kfold=kfold.get_n_splits(df),
                      print_every=args['print_every'])
        
        for i, (train_indices, valid_indices) in enumerate(kfold.split(df)):
            print('Fold:', i)
            df_train = df.loc[train_indices]
            df_valid = df.loc[valid_indices]
            train_data = MedicalData(PATH, df_train, device, transform)
            valid_data = MedicalData(PATH, df_valid, device, transform)
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=64)
            valid_loader = torch.utils.data.DataLoader(valid_data,
                                                       batch_size=64)
            test_loader = None
            trainer.train(train_loader=train_loader,
                          valid_loader=valid_loader,
                          test_loader=test_loader,
                          kfold=kfold)
            print(train_data.class2index)

    else:
        df_train, df_valid, df_test = split_data(PATH, args['sampling'])
        
        trainer = Trainer(model=model,
                      loss=args['loss_function'],
                      epochs=args['num_epochs'],
                      max_lr=args['lr'],
                      device=device,
                      num_samples=len(df_train),
                      print_every=args['print_every'])
        
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
        trainer.train(train_loader=train_loader,
                      valid_loader=valid_loader,
                      test_loader=test_loader)

    if args['load_dir']:
        trainer.load_model(args['load_dir'])

    trainer.train()
    trainer.save_model(args['save_dir'] + args['save_name'])


if __name__ == "__main__":
    main()
