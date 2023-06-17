import os
import glob
import pathlib
import pandas as pd
from datetime import datetime
import yaml
import torch

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

def load_config(config_path):
    stream = open('./config/config.yml', 'r')
    cfg = yaml.load(stream, yaml.Loader)
    return cfg

def create_folder():
    current_time = str(datetime.now())
    save_dir = './save/'
    logs_dir = './logs/'
        
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
    
    return model_dir, visual_dir

def check_gpu(use_gpu):
    return 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'