import os



# from architecture import DecoderTransformer
from builtin_architecture import make_model, make_model_custom
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import init_logger, flush
import torchvision
from trainingmanager import TrainingManager
import torch.nn as nn


def train_model(experiment_directory, epochs, model_params=None, schedule=False, anti=False):
    os.system(f"caffeinate -is -w {os.getpid()} &")

    if model_params is None:
        model_params = {}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainset = get_train_dataset()
    dataloader = get_dataloader(trainset)

    testset = get_test_dataset()
    testloader = get_dataloader(testset)
    if model_params == {}:
        net = make_model()
    else:
        net = make_model_custom(**model_params)
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
    if schedule:
        if anti:
            trainer.train_curriculum(anticurriculum=True)
        else:
            trainer.train_curriculum()
    else:
        trainer.train()
    flush()


if __name__ == "__main__":
    EXPERIMENT_DIRECTORY = "runs/code-decoder-v26-med-scheduled"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True)

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v26-med-unscheduled"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS)

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v26-med-anti"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, anti=True)
