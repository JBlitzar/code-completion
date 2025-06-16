import os
import torch
from logger import log_data, init_logger, log_img
import torch.nn as nn
from tqdm import tqdm, trange
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import numpy as np
from eval import evaluate_topk
from dataset import dataset
from Levenshtein import ratio
from enum import Enum
import signal
import sys

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

        self.clip = 1.0

        self.trainstep_checkin_interval = trainstep_checkin_interval
        self.epochs = epochs

        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        self.net = net
        self.net.to(device)
        self.device = device

        self.dir = dir

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=learning_rate#, weight_decay=1e-5
        )

        # No clue what this does. Maybe its good
        # initialized and never used.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, factor=0.9, patience=10
        )

        self.tracker = ValueTracker()

        self.resume_epoch, self.resume_step = self.get_resume()
        if self.resume_epoch >= self.epochs - 1:
            pass
        elif self.resume_epoch != 0 or self.resume_step != 0:
            self.resume()
        else:
            if os.path.exists(self.dir) and any(
                os.path.isfile(os.path.join(self.dir, item))
                for item in os.listdir(self.dir)
            ):
                raise ValueError(f"The directory '{self.dir}' contains files!")

            os.makedirs(self.dir, exist_ok=True)
            os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)

        print(f"{self.get_param_count()} parameters.")
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        self._interrupted = False

    def _signal_handler(self, signum, frame):
        """Handle keyboard interrupt gracefully"""
        print("\nKeyboard interrupt received. Saving checkpoint...")
        self._interrupted = True

    def _save_on_interrupt(self, epoch, step):
        """Save checkpoint and resume info on interrupt"""
        try:
            self._save("latest.pt")
            self.write_resume(epoch, step)
            print(f"Checkpoint saved at epoch {epoch}, step {step}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        finally:
            print("Exiting...")
            sys.exit(0)

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

    def write_resume(self, epoch, step=0):
        with open(os.path.join(self.dir, "ckpt", "resume.txt"), "w+") as f:
            f.write(f"{epoch},{step}")

    def get_resume(self):
        try:
            with open(os.path.join(self.dir, "ckpt", "resume.txt"), "r") as f:
                content = f.read().strip()
                if ',' in content:
                    epoch, step = content.split(',')
                    return int(epoch), int(step)
                else:
                    # Backward compatibility: if only epoch is stored
                    return int(content), 0
        except (FileNotFoundError, ValueError):
            return 0, 0

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
            print("RESUMING")
            self.resume()

        self._save("latest.pt")  # Just update latest checkpoint
        self.write_resume(epoch, step + 1)  # Save current progress

        log_data(
            {"Loss/Trainstep": self.tracker.average("Loss/trainstep")},
            epoch * dataloader_len + step,
        )
        log_data(
            {"Acc/Trainstep": self.tracker.average("Acc/trainstep")},
            epoch * dataloader_len + step,
        )
        log_data(
            {"TopKAcc/Trainstep": self.tracker.average("TopKAcc/trainstep")},
            epoch * dataloader_len + step,
        )

        self.tracker.reset("Loss/trainstep")
        self.tracker.reset("Acc/trainstep")
        self.tracker.reset("TopKAcc/trainstep")

    def on_epoch_checkin(self, epoch):
        if self.hasnan():
            # revert
            self.resume()

        val_loss = float("inf")
        try:
            val_loss = self.tracker.average("Loss/val/epoch")
        except KeyError:
            pass

        self.save(
            val_loss if val_loss < float("inf") else self.tracker.average("Loss/epoch")
        )

        log_data(
            {
                "Loss/Epoch": self.tracker.average("Loss/epoch"),
                "Loss/Val/Epoch": val_loss,
                "Perplexity/Val/Epoch": float(np.exp(val_loss)),
                "TopKAcc/Epoch": self.tracker.average("TopKAcc/epoch"),
            },
            epoch,
        )

        self.tracker.reset("Acc/epoch")
        self.tracker.reset("Loss/epoch")
        self.tracker.reset("Loss/val/epoch")
        self.tracker.reset("TopKAcc/epoch")
        self.tracker.reset("Perplexity/val/epoch")

        self.write_resume(epoch + 1, 0)  # Start next epoch at step 0

    def eval_model(self, data, compute_metrics=True):
        if type(data) == tuple or type(data) == list:
            data = tuple(d.to(self.device) for d in data)
            batch, attn_mask = data
        else:
            data = data.to(self.device)
            batch = data
            attn_mask = None

        del attn_mask  # unused

        labels = batch[:, 1:].contiguous()
        batch = batch[:, :-1].contiguous()

        # Forward pass
        results = self.net(batch, transpose=True)  # , padding_mask=attn_mask[:, :-1])
        results = results.transpose(0, 1)  # average bug

        # Compute loss
        loss = self.criterion(results.reshape(-1, results.size(-1)), labels.reshape(-1))

        if not compute_metrics:
            return loss, None, None
        
        # Compute accuracy
        preds = results.reshape(-1, results.size(-1)).argmax(dim=1)
        labels_flat = labels.reshape(-1)
        acc = (preds == labels_flat).float().mean()

        # Top-k accuracy
        top_k = 5
        top_k_preds = results.reshape(-1, results.size(-1)).topk(top_k, dim=1).indices
        top_k_acc = (top_k_preds == labels_flat.unsqueeze(1)).any(dim=1).float().mean().item()

        return loss, acc, top_k_acc

    def run_generation(self, data):
        batch, attn_mask = data
        start_sequence = batch[:, :-1].contiguous()[0][:100].unsqueeze(0)
        result = evaluate_topk(
            self.net, start_sequence, amt=100, k=10, temperature=0.8, device=device
        )

        result = dataset.manager.decode(result[0])
        batch_str = dataset.manager.decode(start_sequence[0])

        result = f"<data>{batch_str}</data>{result[len(batch_str):]}"
        # print(result)

        with open(os.path.join(self.dir, "ckpt", "generated.txt"), "a+") as f:
            f.write(f"K=10,T=0.8: {result}\n")

    def epoch_gen(self, loader):
        if loader is not None:
            for data in loader:
                self.run_generation(data)
                break

    def trainstep(self, data):
        self.optimizer.zero_grad()

        loss, acc, topk_acc = self.eval_model(data)

        self.tracker.add("Loss/trainstep", loss.item())
        self.tracker.add("Loss/epoch", loss.item())

        self.tracker.add("Acc/trainstep", acc.item())
        self.tracker.add("TopKAcc/trainstep", topk_acc)
        self.tracker.add("TopKAcc/epoch", topk_acc)

        loss.backward()
        self.optimizer.step()

        return loss.detach(), acc.detach()

    @torch.no_grad()  # decorator yay
    def valstep(self, data):
        loss, acc, topk_acc = self.eval_model(data)

        self.tracker.add("Loss/valstep", loss.item())
        self.tracker.add("Loss/val/epoch", loss.item())

        self.tracker.add("Perplexity/val/epoch", float(np.exp(loss.item())))

        self.tracker.add("TopKAcc/valstep", topk_acc)
        self.tracker.add("TopKAcc/val/epoch", topk_acc)

        return loss.detach(), acc.detach()

    def val_loop(self, val_loader):
        if val_loader is not None:
            for step, data in enumerate(
                test_tqdm := tqdm(
                    val_loader, leave=False, dynamic_ncols=True, desc=f"valloop"
                )
            ):
                self.valstep(data)
                avg_val_loss = self.tracker.average("Loss/val/epoch")
                test_tqdm.set_postfix({"Val Loss": f"{avg_val_loss:.3f}"})

    def train_loop(self, dataloader, epoch):
        start_step = self.resume_step if epoch == self.resume_epoch else 0
        
        for step, data in enumerate(
            train_tqdm := tqdm(
                dataloader, leave=False, dynamic_ncols=True, desc=f"trainloop"
            )
        ):
            # Check for interrupt
            if self._interrupted:
                self._save_on_interrupt(epoch, step)
                raise KeyboardInterrupt("Training interrupted by user")
                
            # Skip steps if resuming
            if step < start_step:
                continue
                
            self.trainstep(data)

            avg_train_loss = self.tracker.average("Loss/trainstep")
            train_tqdm.set_postfix({"Train Loss": f"{avg_train_loss:.3f}"})

            if (
                step % self.trainstep_checkin_interval
                == self.trainstep_checkin_interval - 1
            ):
                
                self.on_trainloop_checkin(epoch, step, len(dataloader))
                

    def epoch(self, epoch: int, dataloader, val_loader=None):
        if self._interrupted:
            return
            
        self.net.train()
        self.train_loop(dataloader, epoch)
        
        if self._interrupted:
            return
            
        tqdm.write(self.get_memory_stats(self.net, dataloader.dataset, sep=" / "))
        self.net.eval()
        self.val_loop(val_loader)

        if self._interrupted:
            return
            
        self.epoch_gen(val_loader)
        self.on_epoch_checkin(epoch)

    def train(self, epochs=None, dataloader=None):

        if epochs is not None:
            self.epochs = epochs

        if dataloader is not None:
            self.dataloader = dataloader

        try:
            for e in trange(
                self.resume_epoch, self.epochs, dynamic_ncols=True, unit_scale=True, unit_divisor=60
            ):
                if self._interrupted:
                    break
                    
                self.epoch(e, self.dataloader, self.val_dataloader)

        except KeyboardInterrupt:
            print("\nTraining interrupted. Checkpoint saved.")
        finally:
            print("Training session ended.")
            gc.collect()
            os.system(
                """osascript -e 'display notification "Training complete" with title "Training Complete"'"""
            )

    @staticmethod
    def get_curriculum_enum():
        return Enum(
            "Curriculum",
            [
                ("NOOP", 1),
                ("CURRICULUM", 2),
                ("ANTICURRICULUM", 3),
                ("SEQUENTIAL", 4),
                ("HYBRID", 5),
            ],
        )

    def train_curriculum(
        self, epochs=None, dataloader=None, curriculum_type=None, loss_based=False
    ):

        print(f"Training curriculum: {curriculum_type} loss_based: {loss_based}")

        Curriculum = self.get_curriculum_enum()

        if curriculum_type is None:
            curriculum_type = Curriculum.NOOP

        if epochs is not None:
            self.epochs = epochs

        if dataloader is not None:
            self.dataloader = dataloader

        sorted_indices = sorted(
            range(len(self.dataloader.dataset)),
            key=lambda i: self.dataloader.dataset[i][1],
            reverse=(curriculum_type.value == Curriculum.ANTICURRICULUM.value),
        )

        # [min(1.0, ((i+1))/epochs) for i in range(epochs)] for normal range
        standard_schedule = [
            min(1.0, ((i + 2) - (i % 2)) / self.epochs) for i in range(self.epochs)
        ]  # [0.2,0.2, 0.4,0.4,0.6,0.6,0.8,0.8,1.0,1.0]
        hybrid_schedule = [
            min(1.0, (i + 2) / self.epochs) for i in range(self.epochs)
        ]  # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0]
        step_size = 1 / (self.epochs / 2)

        try:
            for e in trange(
                self.resume_epoch, self.epochs, dynamic_ncols=True, unit_scale=True, unit_divisor=60
            ):

                if loss_based:
                    sorted_indices = self.get_loss_based_indices(
                        self.dataloader,
                        anti=(curriculum_type.value == Curriculum.ANTICURRICULUM.value),
                    )

                subset_indices = None
                if curriculum_type.value == Curriculum.NOOP.value:
                    print("No curriculum")
                    subset_indices = sorted_indices  # full dataset
                elif curriculum_type.value == Curriculum.SEQUENTIAL.value:
                    print("Sequential curriculum")
                    subset_indices = sorted_indices[
                        int(
                            max(len(sorted_indices) * (standard_schedule[e] - step_size), 0)
                        ) : int(len(sorted_indices) * standard_schedule[e])
                    ]
                elif curriculum_type.value == Curriculum.HYBRID.value:
                    print("Hybrid curriculum")
                    subset_indices = sorted_indices[
                        int(
                            max(len(sorted_indices) * (hybrid_schedule[e] - step_size), 0)
                        ) : int(len(sorted_indices) * hybrid_schedule[e])
                    ]
                elif curriculum_type.value == Curriculum.CURRICULUM.value:
                    print("Curriculum")
                    subset_indices = sorted_indices[
                        : int(len(sorted_indices) * standard_schedule[e])
                    ]
                elif curriculum_type.value == Curriculum.ANTICURRICULUM.value:
                    print("Anti curriculum")
                    subset_indices = sorted_indices[
                        : int(len(sorted_indices) * standard_schedule[e])
                    ]
                else:
                    raise ValueError(f"Unknown curriculum type: {curriculum_type}")

                subset = torch.utils.data.Subset(self.dataloader.dataset, subset_indices)
                cur_dataloader = torch.utils.data.DataLoader(
                    subset, batch_size=self.dataloader.batch_size, shuffle=True#, pin_memory=True
                )

                self.epoch(e, cur_dataloader, self.val_dataloader)

        except KeyboardInterrupt:
            print("\nCurriculum training interrupted. Checkpoint saved.")
        finally:
            print("Curriculum training session ended.")
            gc.collect()
            os.system(
                """osascript -e 'display notification "Training complete" with title "Training Complete"'"""
            )

        print("All done!")
        gc.collect()
        os.system(
            """osascript -e 'display notification "Training complete" with title "Training Complete"'"""
        )

    def get_loss_based_indices(self, dataloader, anti=False):
        losses = []
        # Create a new dataloader with the same dataset but without shuffling
        temp_dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=(
                dataloader.num_workers if hasattr(dataloader, "num_workers") else 0
            ),
        )

        with torch.no_grad():  # Add this for faster inference
            for batch, _ in tqdm(
                temp_dataloader,
                dynamic_ncols=True,
                leave=False,
                desc="Loss-based sorting",
            ):
                loss, _, _ = self.eval_model(batch, compute_metrics=False)
                
                # If the output is a single tensor, convert to list
                if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                    losses.extend([loss.item()] * batch.size(0))
                else:
                    # If the output is already batched
                    losses.extend(loss.detach().cpu().tolist())

        sorted_indices = sorted(
            range(len(dataloader.dataset)), key=lambda i: losses[i], reverse=anti
        )
        return sorted_indices

    def nan_debug(self):
        torch.autograd.set_detect_anomaly(True)

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                return
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"NaNs/Infs detected in {module}")

        for module in self.net.modules():
            module.register_forward_hook(forward_hook)
        self.val_loop(self.val_dataloader)

    def get_param_count(self):
        return sum(p.numel() for p in self.net.parameters())

    def profile_trainstep(self):

        self.net.train()
        data = next(iter(self.dataloader))

        # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("train_step"):
                self.trainstep(data)

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    @staticmethod
    def get_memory_stats(net, trainset, sep="\n"):
        result = ""
        import datetime
        import time
        result += f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + sep
        import psutil
        if torch.backends.mps.is_available():
            result += f"MPS: {torch.mps.current_allocated_memory()/1e9:.2f} GB" + sep
        result += f"RAM: {psutil.virtual_memory().percent}% used" + sep
        
        # Print dataset size
        chunks = getattr(trainset, 'chunks', getattr(trainset.dataset, 'chunks', None))
       
        if chunks is not None:
            result += f"data: {sum(p.numel() * p.element_size() for p in [chunks]) / 1e9:.2f} GB" + sep
        
        # Print model size
        model_size = sum(p.numel() * p.element_size() for p in net.parameters()) / 1e9
        result += f"Params: {model_size:.2f} GB" + sep
        
        # Estimate optimizer size
        optimizer_size = model_size * 2
        result += f"Optim (est): {optimizer_size:.2f} GB" + sep

        return result
