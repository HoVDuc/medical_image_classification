import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm


class Trainer:

    def __init__(self, model, train_loader, valid_loader, test_loader, epochs, learning_rate, device, print_every=5) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = learning_rate
        self.print_every = print_every
        self.optim = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate)
        self.scheduler = OneCycleLR(self.optim, self.lr, self.epochs)
        self.criterion = nn.CrossEntropyLoss()

    def check_device(self):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def training_step(self, batch):
        self.model.train()
        inputs, targets = batch
        pred = self.model(inputs)
        loss = self.criterion(pred, targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def validation(self):
        self.model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                val_inputs, val_targets = batch
                val_pred = self.model(val_inputs)
                val_loss = self.criterion(val_pred, val_targets)
                val_total_loss += val_loss
            avg_val_loss = val_total_loss / len(self.valid_loader)
        
        return avg_val_loss
        
    def train(self):
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            pbar = tqdm(self.train_loader, ncols=100,
                        desc="EPOCH: {}/{}".format(epoch, self.epochs))
            for batch in pbar:
                loss = self.training_step(batch)
                pbar.set_description('loss: {} - optimizer: {}'.format(loss, self.scheduler.get_lr()))
                total_loss += loss
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            
            if epoch % self.print_every == 0:
                avg_val_loss = self.validation()
                print("loss: {} - val_loss: {}".format(avg_loss, avg_val_loss))
                    