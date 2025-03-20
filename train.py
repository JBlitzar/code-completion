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


def train_model(experiment_directory, epochs, model_params=None, schedule=False, **kwargs):
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
        trainer.train_schedule(**kwargs)
    else:
        trainer.train()
    flush()


if __name__ == "__main__":
    del_runs = True
    if del_runs:
        del_runs = del_runs and input("Confirm that this will delete checkpoints") == "y"
        if not del_runs:
            print("Exiting")
            exit()

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/noop"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, noop=True)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/curriculum-noloss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, curriculum=True, loss_based=False)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/curriculum-loss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, curriculum=True, loss_based=True)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/anticurriculum-noloss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, anticurriculum=True, loss_based=False)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/anticurriculum-loss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, anticurriculum=True, loss_based=True)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/sequential-noloss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, sequential=True, loss_based=False)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/sequential-loss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, sequential=True, loss_based=True)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/hybrid-noloss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, hybrid=True, loss_based=False)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")

    EXPERIMENT_DIRECTORY = "runs/code-decoder-v27-alltrains-experiment/hybrid-loss"
    EPOCHS = 10
    train_model(EXPERIMENT_DIRECTORY, EPOCHS, schedule=True, hybrid=True, loss_based=True)
    if del_runs:
        os.system(f"rm -r {EXPERIMENT_DIRECTORY}/ckpt/*.pt")
