import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from loss.focal_loss import FocalLoss
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.metrics import classification_report
import itertools


class Trainer:

    def __init__(self, model, num_classes, loss, epochs, max_lr, device, num_samples, kfold=1, print_every=5) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = max_lr
        self.print_every = print_every
        loss_func = {
            'ce': nn.CrossEntropyLoss(),
            'focal': FocalLoss(gamma=2.0)
        }
        self.criterion = loss_func[loss]
                
        # Optimizer
        self.optim = torch.optim.Adam(params=self.model.parameters())
        self.scheduler = OneCycleLR(optimizer=self.optim, 
                                    max_lr=self.lr, 
                                    total_steps=self.epochs*num_samples*kfold)
        
        self.writer = SummaryWriter('runs/Medical/trying_tensorboard')

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
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler.step()
        return loss.item()

    def f1_scores(self, preds, targets):
        f1 = torchmetrics.F1Score(
            task='multiclass', num_classes=self.num_classes).to(self.device)
        return f1(preds, targets)

    def precision(self, preds, targets):
        metric = MulticlassPrecision(num_classes=self.num_classes).to(self.device)
        return metric(preds, targets)

    def recall(self, preds, targets):
        metric = MulticlassRecall(num_classes=self.num_classes).to(self.device)
        return metric(preds, targets)

    def mean_average_precision(self, preds, targets):
        average_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=self.num_classes, average="macro", thresholds=5).to(self.device)
        ap = average_precision(preds, targets)
        return ap

    def confusion_matrix(self, preds, targets):
        metric = MulticlassConfusionMatrix(num_classes=self.num_classes).to(self.device)
        return metric(preds, targets)
    
    def metrics(self, pred, targets):
        f1_scores = self.f1_scores(pred, targets)
        mAP = self.mean_average_precision(pred, targets)
        precision = self.precision(pred, targets)
        recall = self.recall(pred, targets)
        confusion_matrix = self.confusion_matrix(pred, targets)
        return f1_scores, mAP, precision, recall, confusion_matrix
    
    def display_classification_report(preds, targets):
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
            'mAP': 0,
            'recall': 0
        }
        total_cf = torch.zeros(torch.Size([self.num_classes, self.num_classes]), device=self.device)
        data_loader = self.valid_loader if mode == 'val' else self.test_loader
        n_sample = len(data_loader)
        preds, targets = [], []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                loss, pred, target = self.validation_step(batch)
                f1_scores, mAP, precision, recall, confusion_matrix = self.metrics(pred, target)
                total_metrics['loss'] += loss
                total_metrics['f1_scores'] += f1_scores
                total_metrics['precision'] += precision
                total_metrics['mAP'] += mAP
                total_metrics['recall'] += recall
                total_cf += confusion_matrix

                preds.append(torch.argmax(pred, dim=1).cpu().tolist())
                targets.append(target.cpu().tolist())
            
            preds = list(itertools.chain(*preds))
            targets = list(itertools.chain(*targets))

            self.display_classification_report(preds, targets)
            avg = {metric: total_metrics[metric] / n_sample for metric in total_metrics}
        return avg, total_cf

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print('Saved!')

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        print('Loaded!')

    def train(self, train_loader, valid_loader, test_loader, kfold=1):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        step = 0
        for epoch in range(1, self.epochs + 1):
            print("EPOCH: {}/{}".format(epoch, self.epochs))
            print('---' * 20)
            total_loss = 0
            pbar = tqdm(self.train_loader, ncols=100)
            for batch in pbar:
                loss = self.training_step(batch)
                pbar.set_description('loss: {}'.format(loss))
                total_loss += loss
            avg_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar('Training loss', avg_loss, global_step=step)
            step += 1
            if epoch % self.print_every == 0:
                avg_val_metrics, cf = self.validation(
                    mode='val')

                print('Avg loss:', avg_loss)
                print('metrics:', avg_val_metrics)
                print('Confusion Matrix:\n', cf)

        if self.test_loader:
            avg_test_metrics, cf = self.validation(mode='test')
            print('metrics test:', avg_test_metrics)
            print('Confusion Matrix:\n', cf)
