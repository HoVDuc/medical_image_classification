import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from loss.focal_loss import FocalLoss


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
        self.criterion = FocalLoss(gamma=2.0, alpha=0.2)

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
    
    def mean_average_precision(self, preds, targets):
        average_precision = torchmetrics.AveragePrecision(task="multiclass", num_classes=6, average=None).to(self.device)
        ap = average_precision(preds, targets)
        mAP = 1/6 * sum(ap)
        return mAP

    def validation(self):
        self.model.eval()
        val_total_loss = 0
        # val_total_accuracy = 0
        val_total_f1_scores = 0
        val_total_map = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                val_inputs, val_targets = batch
                val_pred = self.model(val_inputs)
                val_loss = self.criterion(val_pred, val_targets)
                val_total_loss += val_loss
                accuracy = self.calc_accuracy(val_pred, val_targets)
                f1_scores = self.f1_scores(val_pred, val_targets)
                mAP = self.mean_average_precision(val_pred, val_targets)
                # val_total_accuracy += accuracy
                val_total_f1_scores += f1_scores
                val_total_map += mAP

            avg_val_loss = val_total_loss / len(self.valid_loader)
            # avg_val_accuracy = val_total_accuracy / len(self.valid_loader)
            avg_val_f1_scores = val_total_f1_scores / len(self.valid_loader)
            avg_val_map = val_total_map / len(self.valid_loader)

        return avg_val_loss, avg_val_f1_scores, avg_val_map
    
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
                avg_val_loss, avg_val_f1_scores, avg_val_map = self.validation()

                print("loss: {} - val_loss: {} - f1_scores: {} - mAP: {}".format(
                    avg_loss, avg_val_loss, avg_val_f1_scores, avg_val_map))
