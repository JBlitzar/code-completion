import os


# from architecture import DecoderTransformer
from builtin_architecture import make_model, make_model_custom
from dataset import fromDataset, get_dataloader, TextCorpusDataset
import torch
from tqdm import tqdm, trange
from logger import init_logger, flush
import torchvision
from trainingmanager import TrainingManager
import torch.nn as nn


def train_model(
    experiment_directory,
    trainset,
    testset,
    epochs,
    model_params=None,
    schedule=False,
    **kwargs,
):
    os.system(f"caffeinate -is -w {os.getpid()} &")

    if model_params is None:
        model_params = {}

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataloader = get_dataloader(trainset)

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
        trainer.train_curriculum(**kwargs)
    else:
        trainer.train()
    flush()


def run_experiment(experiment_directory, epochs, trainset, testset, del_runs, **kwargs):
    train_model(experiment_directory, trainset, testset, epochs, schedule=True, **kwargs)
    if del_runs:
        os.system(f"rm -r {experiment_directory}/ckpt/*.pt")


if __name__ == "__main__":
    del_runs = True
    if del_runs:
        del_runs = (
            del_runs and input("Confirm that this will delete checkpoints: ") == "y"
        )
        if not del_runs:
            print("Exiting")
            exit()

    parent_directory = "runs/code-decoder-v30-alltrains-v3"

    Curriculum = TrainingManager.get_curriculum_enum()

    experiments = [
        (
            "curriculum-loss",
            {"curriculum_type": Curriculum.CURRICULUM, "loss_based": True},
        ),
        ("noop", {"curriculum_type": Curriculum.NOOP, "loss_based": False}),
        (
            "curriculum-noloss",
            {"curriculum_type": Curriculum.CURRICULUM, "loss_based": False},
        ),
        (
            "anticurriculum",
            {"curriculum_type": Curriculum.ANTICURRICULUM, "loss_based": False},
        ),
        (
            "anticurriculum-loss",
            {"curriculum_type": Curriculum.ANTICURRICULUM, "loss_based": True},
        ),
        ("sequential", {"curriculum_type": Curriculum.SEQUENTIAL, "loss_based": False}),
        (
            "sequential-loss",
            {"curriculum_type": Curriculum.SEQUENTIAL, "loss_based": True},
        ),
        ("hybrid", {"curriculum_type": Curriculum.HYBRID, "loss_based": False}),
        ("hybrid-loss", {"curriculum_type": Curriculum.HYBRID, "loss_based": True}),
    ]

    EPOCHS = 10
    for experiment_name, params in experiments:
        experiment_directory = os.path.join(parent_directory, experiment_name)
        trainset, testset = fromDataset(
            TextCorpusDataset(
                root_dir=os.path.expanduser(
                    "~/torch_datasets/github-python/all_trains_subset_corpus"
                ),
                vocab_size=153127,
                IS_CODE=True,
                IS_CUSTOM=True,
                max_length=256,
                sliding_window=False,
                stride=10,
                get_rarity_score=True,
                get_entropy_score=False # change to True and change the above to false for entropy score instead
            )
        )
        print(f"Running experiment: {experiment_name}")
        print(f"Params: {params}")
        print(len(trainset), len(testset))
        print(trainset[3])

        run_experiment(
            experiment_directory,
            EPOCHS,
            trainset,
            testset,
            del_runs,
            **params,
        )
