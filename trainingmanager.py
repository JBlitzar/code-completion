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
    def __init__(
        self,
        net: nn.Module,
        dir: str,
        dataloader,
        device=device,
        trainstep_checkin_interval=100,
        epochs=100,
        val_dataloader=None,
    ):

        learning_rate = 0.001

        self.trainstep_checkin_interval = trainstep_checkin_interval
        self.epochs = epochs

        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        self.net = net
        self.net.to(device)
        self.device = device

        self.dir = dir

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.tracker = ValueTracker()

        self.resume_amt = self.get_resume()
        if self.resume_amt != 0:
            self.resume()
        else:
            if os.path.exists(self.dir) and any(
                os.path.isfile(os.path.join(self.dir, item))
                for item in os.listdir(self.dir)
            ):
                raise ValueError(f"The directory '{self.dir}' contains files!")

            os.makedirs(self.dir, exist_ok=True)
            os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)

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
        self.net.load_state_dict(
            torch.load(os.path.join(self.dir, "ckpt", name), weights_only=True)
        )

    def write_resume(self, epoch):
        with open(os.path.join(self.dir, "ckpt", "resume.txt"), "w+") as f:
            f.write(str(epoch))

    def get_resume(self):
        try:
            with open(os.path.join(self.dir, "ckpt", "resume.txt"), "r") as f:
                return int(f.read())
        except (FileNotFoundError, ValueError):
            return 0

    def write_best_val_loss(self, loss):
        with open(os.path.join(self.dir, "ckpt", "best_val_loss.txt"), "w+") as f:
            f.write(f"{loss:.6f}")

    def get_best_val_loss(self):
        try:
            with open(os.path.join(self.dir, "ckpt", "best_val_loss.txt"), "r") as f:
                return float(f.read())
        except (FileNotFoundError, ValueError):
            return float("inf")

    def resume(self):
        self._load("latest.pt")

    def save(self, loss):
        self._save("latest.pt")

        best_val_loss = self.get_best_val_loss()
        if loss < best_val_loss:
            best_val_loss = loss
            self._save("best.pt")
            self.write_best_val_loss(best_val_loss)

        # self._save(f"{prefix}_{step}.pt")

    def on_trainloop_checkin(self, epoch, step, dataloader_len):
        if self.hasnan():
            # revert
            self.resume()

        self._save("latest.pt")  # Just update latest checkpoint

        log_data(
            {"Loss/Trainstep": self.tracker.average("Loss/trainstep")},
            epoch * dataloader_len + step,
        )

        self.tracker.reset("Loss/trainstep")

    def on_epoch_checkin(self, epoch):
        if self.hasnan():
            # revert
            self.resume()

        val_loss = float("inf")
        try:
            val_loss = self.tracker.average("Loss/val/epoch")
        except KeyError:
            pass

        self.save(val_loss if val_loss < float("inf") else self.tracker.average("Loss/epoch"))

        log_data(
            {
                "Loss/Epoch": self.tracker.average("Loss/epoch"),
                "Loss/Val/Epoch": val_loss,
            },
            epoch,
        )

        self.tracker.reset("Loss/epoch")
        self.tracker.reset("Loss/val/epoch")

        self.write_resume(epoch)

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

    @torch.no_grad()  # decorator yay
    def valstep(self, data):

        data = tuple(d.to(self.device) for d in data)

        self.optimizer.zero_grad()

        # Different for every model
        batch, attn_mask = data

        labels = batch[:, 1:].contiguous()
        batch = batch[:, :-1].contiguous()

        results = self.net(batch, padding_mask=attn_mask[:, :-1])

        loss = self.criterion(results.view(-1, results.size(-1)), labels.view(-1))

        # self.tracker.add("Loss/valstep", loss.item())
        self.tracker.add("Loss/val/epoch", loss.item())

    def epoch(self, epoch: int, dataloader, val_loader=None):
        for step, data in enumerate(tqdm(dataloader, leave=False, dynamic_ncols=True)):
            self.trainstep(data)

            if (
                step % self.trainstep_checkin_interval
                == self.trainstep_checkin_interval - 1
            ):
                self.on_trainloop_checkin(epoch, step, len(dataloader))

        if val_loader is not None:
            for step, data in enumerate(
                tqdm(val_loader, leave=False, dynamic_ncols=True)
            ):
                self.valstep(data)

        self.on_epoch_checkin(epoch)

    def train(self, epochs=None, dataloader=None):

        if epochs is not None:
            self.epochs = epochs

        if dataloader is not None:
            self.dataloader = dataloader

        for e in trange(self.epochs, dynamic_ncols=True):

            if e <= self.resume_amt:
                continue

            self.epoch(e, self.dataloader, self.val_dataloader)
