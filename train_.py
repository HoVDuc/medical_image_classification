import os
import sys
import glob
import pathlib
import yaml
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision
import argparse
from sklearn.model_selection import KFold, train_test_split
from torchvision import transforms
from dataloader import MedicalData
from trainer import Trainer
from datetime import datetime

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

    df = pd.DataFrame(data)

    if not os.path.isfile('./class.csv'):
        data_classes = {
            'class': []
        }
        for i, label in enumerate(set(df['label'])):
            data_classes['class'].append(label)

        df_class = pd.DataFrame(data_classes)
        df_class.to_csv('./class.csv', index=False)

    return df

def undersampling(df):
    df_list = []
    for label in set(df['label'].values):
        sampling = df[df['label'] == label].head(
            df['label'].value_counts()['AIS']).reset_index(drop=True)
        df_list.append(sampling)

    df_sampling = pd.concat(df_list, axis=0).reset_index(drop=True)
    return df_sampling

def kfold_split(PATH, n_splits=5):
    df = load_data(PATH)
    df = undersampling(df)
    df_sample = df.sample(frac=1).reset_index(drop=True)
    kfold = KFold(n_splits=n_splits)
    return df_sample[:100], kfold

def check_gpu(use_gpu):
    return 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'

def main():
    parser = argparse.ArgumentParser(description='Classification training')
    parser.add_argument('--save_dir', type=str, default='./save/')
    parser.add_argument('--save_name', type=str, default='model.pt')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--logs_dir', type=str, default='./logs/')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    current_time = str(datetime.now())
    stream = open('./config/config.yml', 'r')
    cfg = yaml.load(stream, yaml.Loader)
    
    PATH = cfg['datasets']['root']
    batch_size = cfg['datasets']['batch_size']
    img_width = cfg['datasets']['image_width']
    img_height = cfg['datasets']['image_height']
    save_dir = args['save_dir']
    logs_dir = args['logs_dir']
        
    model_dir = os.path.join(save_dir, 'save_{}'.format(current_time))
    try:
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
    except FileNotFoundError:
        os.mkdir(save_dir)
        os.mkdir(model_dir)
    
    visual_dir = os.path.join(logs_dir, 'logs_{}'.format(current_time))
    if not os.path.isdir(visual_dir):
        try:
            os.mkdir(visual_dir)
        except FileNotFoundError:
            os.mkdir(logs_dir)
            os.mkdir(visual_dir)

    device = check_gpu(cfg['use_gpu'])

    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.GaussianBlur(3),
        transforms.RandomHorizontalFlip(0.3),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])

    classes = pd.read_csv('./class.csv')
    cfg_model = cfg['model']
    
    assert cfg_model != len(classes), 'number classes not fit'
    model = timm.create_model(
        cfg_model['name'], pretrained=cfg_model['pretrained'], num_classes=cfg_model['num_classes'])

    df_train = pd.read_csv('./data/train_set.csv')
    df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=42)
    df_test = pd.read_csv('./data/test_set.csv')
    
    train_data = MedicalData(
        PATH, df_train, classes, device, transform)
    valid_data = MedicalData(
        PATH, df_valid, classes, device, transform)
    test_data = MedicalData(
        PATH, df_test, classes, device, transform)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                                batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size)
    
    trainer = Trainer(model=model, 
                        config=cfg,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        test_loader=test_loader,
                        device=device)
    
    if args['load_dir']:
        trainer.load_model(args['load_dir'])
    
    trainer.train(kfold=1, save_dir=model_dir)
    trainer.visualize(visual_dir, 1)
    print(train_data.class2index)

    
if __name__ == "__main__":
    main()
