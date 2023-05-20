import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from loss.focal_loss import FocalLoss
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall


class Trainer:

    def __init__(self, model, train_loader, valid_loader, test_loader, epochs, max_lr, device, print_every=5) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = max_lr
        self.print_every = print_every
        self.optim = torch.optim.Adam(
            params=self.model.parameters())
        self.scheduler = OneCycleLR(self.optim, self.lr, self.epochs)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss(gamma=2.0)

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
        return loss.item()

    def f1_scores(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        f1 = torchmetrics.F1Score(task='multiclass', num_classes=6).to(self.device)
        return f1(preds, targets)
    
    def precision(self, preds, targets):
        metric = MulticlassPrecision(num_classes=6)
        return metric(preds, targets)
    
    def recall(self, preds, targets):
        metric = MulticlassRecall(num_classes=6)
        return metric(preds, targets)
    
    def mean_average_precision(self, preds, targets):
        average_precision = torchmetrics.AveragePrecision(task="multiclass", num_classes=6, average="macro", thresholds=5).to(self.device)
        ap = average_precision(preds, targets)
        mAP = 1/6 * sum(ap)
        return mAP

    def validation(self, mode='val'):
        self.model.eval()
        total_loss = 0
        # total_accuracy = 0
        total_f1_scores = 0
        total_map = 0
        total_precision = 0
        total_recall = 0
        
        data_loader = self.valid_loader if mode == 'val' else self.test_loader
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                pred = self.model(inputs)
                loss = self.criterion(pred, targets)
                total_loss += loss
                # calc metrics
                # accuracy = self.calc_accuracy(pred, targets)
                f1_scores = self.f1_scores(pred, targets)
                mAP = self.mean_average_precision(pred, targets)
                precision = self.precision(pred, targets)
                recall = self.recall(pred, targets)
                # calc total metrics
                # total_accuracy += accuracy
                total_f1_scores += f1_scores
                total_map += mAP
                total_precision += precision
                total_recall += recall
                
            avg_loss = total_loss / len(data_loader)
            # avg_accuracy = total_accuracy / len(data_loader)
            avg_f1_scores = total_f1_scores / len(data_loader)
            avg_map = total_map / len(data_loader)
            avg_precision = total_precision / len(data_loader)
            avg_recall = total_recall / len(data_loader)

        return avg_loss, avg_f1_scores, avg_map, avg_precision, avg_recall
    
    def train(self):
        for epoch in range(1, self.epochs + 1):
            print("EPOCH: {}/{}".format(epoch, self.epochs))
            print('---' * 20)
            total_loss = 0
            pbar = tqdm(self.train_loader, ncols=100)
            for batch in pbar:
                loss = self.training_step(batch)
                pbar.set_description('loss: {}'.format(loss))
                total_loss += loss
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)

            if epoch % self.print_every == 0:
                avg_val_loss, avg_val_f1_scores, avg_val_map, avg_val_precision, avg_val_recall = self.validation(mode='val')

                print("loss: {} - val_loss: {} - f1_scores: {} - mAP: {} - precision: {} - recall: {}".format(
                    avg_loss, avg_val_loss, avg_val_f1_scores, avg_val_map, avg_val_precision, avg_val_recall))
        avg_test_loss, avg_test_f1_scores, avg_test_map, avg_test_precision, avg_test_recall = self.validation(mode='test')
        print("test_loss: {} - f1_scores: {} - mAP: {} - precision: {}: recall: {}".format(
                        avg_test_loss, avg_test_f1_scores, avg_test_map, avg_test_precision, avg_test_recall))
