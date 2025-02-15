import os

os.system(f"caffeinate -is -w {os.getpid()} &")

# from architecture import DecoderTransformer
from builtin_architecture import make_model
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import init_logger
import torchvision
from trainingmanager import TrainingManager
import torch.nn as nn

def train_model(experiment_directory, epochs):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainset = get_train_dataset()
    dataloader = get_dataloader(trainset)

    testset = get_test_dataset()
    testloader = get_dataloader(testset)

    net = make_model()
    net.to(device)

    trainer = TrainingManager(
        net=net,
        dir=experiment_directory,
        dataloader=dataloader,
        device=device,
        trainstep_checkin_interval=100,
        epochs=epochs,
        val_dataloader=testloader,
    )

    for batch, attn_mask in dataloader:
        init_logger(
            net,
            dir=os.path.join(experiment_directory, "tensorboard"),
        )
        break

    trainer.train()

if __name__ == "__main__":
    EXPERIMENT_DIRECTORY = "runs/code-decoder-v22-bigset-tuner"
    EPOCHS = 50
    train_model(EXPERIMENT_DIRECTORY, EPOCHS)
