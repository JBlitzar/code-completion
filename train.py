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


EXPERIMENT_DIRECTORY = "runs/code-decoder-v22-bigset-tuner"


device = "mps" if torch.backends.mps.is_available() else "cpu"
trainset = get_train_dataset()
dataloader = get_dataloader(trainset)

testset = get_test_dataset()
testloader = get_dataloader(testset)


net = (
    make_model()
)  # nn.Transformer(d_model=128, nhead=1, num_decoder_layers=2, num_encoder_layers=0)#DecoderTransformer(vocab_size=199, num_blocks=1)
net.to(device)


trainer = TrainingManager(
    net=net,
    dir=EXPERIMENT_DIRECTORY,
    dataloader=dataloader,
    device=device,
    trainstep_checkin_interval=100,
    epochs=20,
    val_dataloader=testloader,
)

# trainer.profile_trainstep()

for batch, attn_mask in dataloader:
    init_logger(
        net,
        # batch.to(device),#, attn_mask.to(device)),
        dir=os.path.join(EXPERIMENT_DIRECTORY, "tensorboard"),
    )
    break


trainer.train()
# os.system("bash cleanup.sh")
