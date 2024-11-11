import os
os.system(f"caffeinate -is -w {os.getpid()} &")

from architecture import Transformer
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import init_logger
import torchvision
from transformers import AutoTokenizer
from trainingmanager import TrainingManager




tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')




EXPERIMENT_DIRECTORY = "runs/shakespeare-test-v2"





device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset())

testloader = get_dataloader(get_test_dataset())


net = Transformer()
net.to(device)


trainer = TrainingManager(
    net=net,
    dir=EXPERIMENT_DIRECTORY,
    dataloader=dataloader,
    device=device,
    trainstep_checkin_interval=100,
    epochs=100
)




for batch, attn_mask in dataloader:
    init_logger(net, (batch.to(device), attn_mask.to(device)), dir=os.path.join(EXPERIMENT_DIRECTORY, "tensorboard"))
    break

trainer.train()

