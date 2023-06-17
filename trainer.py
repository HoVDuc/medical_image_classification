import os
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.notebook import tqdm
from loss.focal_loss import FocalLoss
from loss.class_balanced_loss import ClassBalanced
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import pandas as pd
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
import numpy as np


class Trainer:

    def __init__(self, model, config, train_loader, valid_loader, test_loader, device) -> None:
        self.config = config
        self.epochs = config['trainer']['epochs']
        self.device = device
        self.model = model.to(self.device)
        self.num_classes = config['model']['num_classes']
        self.lr = float(config['trainer']['lr'])
        self.max_lr = float(config['trainer']['max_lr'])
        self.print_every = config['trainer']['print_every']
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        loss_func = {
            'ce': nn.CrossEntropyLoss(),
            'focal': FocalLoss(gamma=2.0),
            'cb': ClassBalanced(self.num_classes)
        }
        self.criterion = loss_func[config['loss']['name']]

        # Optimizer
        self.optim = torch.optim.AdamW(
            params=self.model.parameters(), lr=self.lr)
        self.scheduler = OneCycleLR(optimizer=self.optim,
                                    max_lr=self.max_lr,
                                    total_steps=self.epochs*len(train_loader)*int(config['kfold']['num_fold']))

    def check_device(self):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def calc_accuracy(self, pred, targets):
        pred = torch.argmax(pred, dim=1)
        accuracy = sum(pred == targets)
        return accuracy / len(targets)

    def training_step(self, batch):
        self.model.train()
        inputs, targets = batch
        pred = self.model(inputs)
        loss = self.criterion(pred, targets)
        reg_loss = self.regularization_loss(lamda=1e-3, model=self.model, l=2)
        loss += reg_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler.step()

        return loss.item(), self.accuracy(pred, targets).item()

    def regularization_loss(self, lamda, model, l=2):
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.norm(param, p=l)
        return lamda * reg_loss

    def confusion_matrix(self, preds, targets):
        metric = MulticlassConfusionMatrix(
            num_classes=self.num_classes).to(self.device)
        return metric(preds, targets)

    def calc_metrics(self, pred, targets):

        def f1_scores(self, preds, targets):
            f1 = torchmetrics.F1Score(
                task='multiclass', num_classes=self.num_classes).to(self.device)
            return f1(preds, targets)

        def precision(self, preds, targets):
            metric = MulticlassPrecision(
                num_classes=self.num_classes).to(self.device)
            return metric(preds, targets)

        def recall(self, preds, targets):
            metric = MulticlassRecall(num_classes=self.num_classes).to(self.device)
            return metric(preds, targets)
        
        f1_scores = self.f1_scores(pred, targets)
        precision = self.precision(pred, targets)
        recall = self.recall(pred, targets)
        confusion_matrix = self.confusion_matrix(pred, targets)
        return f1_scores, precision, recall, confusion_matrix

    def display_classification_report(self, preds, targets):
        targets_name = pd.read_csv('./class.csv')['class'].to_list()
        print(classification_report(targets, preds, target_names=targets_name))

    def validation_step(self, batch):
        self.model.eval()
        inputs, targets = batch
        pred = self.model(inputs)
        loss = self.criterion(pred, targets)

        return loss.item(), pred, targets

    def validation(self, mode='val'):

        total_metrics = {
            'loss': 0,
            'f1_scores': 0,
            'precision': 0,
            'recall': 0
        }
        total_cf = torch.zeros(torch.Size(
            [self.num_classes, self.num_classes]), device=self.device)
        data_loader = self.valid_loader if mode == 'val' else self.test_loader
        n_sample = len(data_loader)
        preds, targets = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                loss, pred, target = self.validation_step(batch)
                f1_scores, precision, recall, confusion_matrix = self.metrics(pred, target)
                total_metrics['loss'] += np.round(loss, 3)
                total_metrics['f1_scores'] += np.round(f1_scores.item(), 3)
                total_metrics['precision'] += np.round(precision.item(), 3)
                total_metrics['recall'] += np.round(recall.item(), 3)
                total_cf += confusion_matrix
                preds.append(torch.argmax(pred, dim=1).cpu().tolist())
                targets.append(target.cpu().tolist())

            preds = list(itertools.chain(*preds))
            targets = list(itertools.chain(*targets))

            self.display_classification_report(preds, targets)
            avg = {metric: total_metrics[metric] /
                   n_sample for metric in total_metrics}
        return avg, total_cf

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print('Saved!')

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        print('Loaded!')

    def train(self, kfold, save_dir):
        for epoch in range(1, self.epochs + 1):
            print("EPOCH: {}/{}".format(epoch, self.epochs))
            total_loss = 0
            total_accuracy = 0
            n_batch = len(self.train_loader)
            pbar = tqdm(self.train_loader)
            for i, batch in enumerate(pbar):
                loss, accuracy = self.training_step(batch)
                total_loss += loss
                pbar.set_description("Loss: {}".format(total_loss / (i+1)))
                total_accuracy += accuracy

            if epoch % self.print_every == 0:
                avg_val_metrics, cf = self.validation(
                mode='val')
                # print('Avg loss:', avg_loss)
                print('metrics:', avg_val_metrics)
                print('Confusion Matrix:\n', cf)

            save_path = os.path.join(
                save_dir, 'model_fold{}_epoch{}.pt'.format(kfold, epoch))
            self.save_model(save_path)

        if self.test_loader:
            avg_test_metrics, cf = self.validation(mode='test')
            print('metrics test:', avg_test_metrics)
            print('Confusion Matrix:\n', cf)

    def visualize(self, logs_dir, kfold):
        def visual(type='loss'):
            path_dir = os.path.join(logs_dir, '{}/'.format(type))
            if not os.path.isdir(path_dir):
                os.mkdir(path_dir)

            val_key = 'val_{}'.format(type)
            plt.plot(self.history[type], label=type)
            plt.plot(self.history[val_key], label=val_key)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(
                path_dir, 'fig_fold_{}.jpg'.format(kfold)))
            plt.clf()
            print('Saveed! in ', logs_dir)

        visual('loss')
        visual('acc')
        plt.show()
