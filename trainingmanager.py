import os
import torch
from logger import log_data, init_logger, log_img
import torch.nn as nn
from tqdm import tqdm, trange


device = "mps" if torch.backends.mps.is_available() else "cpu"


from collections import defaultdict

class ValueTracker:
    def __init__(self):
        self.data = {}
    
    def add(self, label, value):
        if label not in self.data:
            self.data[label] = []
        self.data[label].append(value)
    
    def average(self, label):
        values = self.data[label]
        if values:
            return sum(values) / len(values)
        else:
            return 0.0
    
    def reset(self, label=None):
        if label is not None:
            if label in self.data:
                self.data[label] = []
        else:
            self.data = {}

    
    def get_values(self, label):
        return self.data[label]
    

    def summary(self):
        for label in self.data:
            avg = self.average(label)
            print(f"{label} - Average: {avg:.4f}")



class TrainingManager:
    def __init__(self, net: nn.Module, dir: str, dataloader, device=device, trainstep_checkin_interval=100, epochs=100):
        
        learning_rate=0.001


        self.trainstep_checkin_interval = trainstep_checkin_interval
        self.epochs = epochs

        self.dataloader = dataloader


        self.net = net
        self.net.to(device)
        self.device = device

        self.dir = dir

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)


        self.tracker = ValueTracker()

    def hasnan(self):
        for _, param in self.net.named_parameters():
            if torch.isnan(param).any():
                return True
        for _, param in self.net.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return True
            
        return False

    def _save(self, name="latest.pt"):
        with open(os.path.join(self.dir, "ckpt", name), "wb+") as f:
            torch.save(self.net.state_dict(), f)

    def _load(self, name="latest.pt"):
        self.net.load_state_dict(torch.load(os.path.join(self.dir, "ckpt", name), weights_only=True))

    def resume(self):
        self.load("latest.pt")

    def save(self, step, prefix="epoch"):
        self._save(f"{prefix}_{step}.pt")
        self._save("latest.pt")


    def on_trainloop_checkin(self, epoch, step, dataloader_len):
        if self.hasnan():
            #revert
            self.resume()
        
        self._save("latest.pt") # Just update latest checkpoint

        log_data({
        "Loss/Trainstep":self.tracker.average("Loss/trainstep")
        },epoch * dataloader_len + step)

        self.tracker.reset("Loss/trainstep")

    def on_epoch_checkin(self, epoch):
        if self.hasnan():
            #revert
            self.resume()
        
        self.save(epoch, "epoch")

        log_data({
        "Loss/Epoch":self.tracker.average("Loss/epoch")
        },epoch)

        self.tracker.reset("Loss/epoch")

    def trainstep(self, data):
        

        data = tuple(d.to(self.device) for d in data)

        self.optimizer.zero_grad()
        
        # Different for every model
        batch, attn_mask = data

        labels = batch[:, 1:].contiguous()
        batch = batch[:, :-1].contiguous()

        results = self.net(batch, padding_mask=attn_mask[:, :-1])

        loss = self.criterion(results.view(-1, results.size(-1)), labels.view(-1))

        loss.backward()

        self.optimizer.step()

        
        self.tracker.add("Loss/trainstep", loss.item())
        self.tracker.add("Loss/epoch", loss.item())


    def epoch(self, epoch: int, dataloader):
        for step, data in tqdm(enumerate(dataloader), leave=False):
            self.trainstep(data)

            
            if step % self.trainstep_checkin_interval == self.trainstep_checkin_interval - 1:
                self.on_trainloop_checkin(epoch, step, len(dataloader))

        self.on_epoch_checkin(epoch)


    def train(self,epochs = None,dataloader=None):

        if epochs is not None:
            self.epochs = epochs

        if dataloader is not None:
            self.dataloader = dataloader

        for e in trange(self.epochs):
            self.epoch(e,self.dataloader)
    
    