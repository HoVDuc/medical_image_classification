import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:

    def __init__(self, model, train_loader, valid_loader, epochs, learning_rate, device) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.lr = learning_rate

        self.optim = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate)
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

    def validation_step(self):
        pass

    def train(self):
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            pbar = tqdm(self.train_loader, ncols=100,
                        desc="EPOCH: {}/{}".format(epoch, self.epochs))
            for batch in pbar:
                loss = self.training_step(batch)
                total_loss += loss
            pbar.set_description('loss: {}'.format(
                total_loss / len(self.train_loader)))
