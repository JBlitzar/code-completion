import os
os.system(f"caffeinate -is -w {os.getpid()} &")

from architecture import Transformer
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import log_data, init_logger, log_img
import torchvision
from transformers import AutoTokenizer


RESUME = 3





tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')




EXPERIMENT_DIRECTORY = "runs/run1-python"
if RESUME == 0:
    if os.path.exists(EXPERIMENT_DIRECTORY) and any(os.path.isfile(os.path.join(EXPERIMENT_DIRECTORY, item)) for item in os.listdir(EXPERIMENT_DIRECTORY)):
        raise ValueError(f"The directory '{EXPERIMENT_DIRECTORY}' contains files, not just subfolders!")

    os.makedirs(EXPERIMENT_DIRECTORY, exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_DIRECTORY, "ckpt"), exist_ok=True)





device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset())

testloader = get_dataloader(get_test_dataset())


net = Transformer()
net.to(device)
if RESUME != 0:
    net.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIRECTORY, "ckpt/latest.pt"), weights_only=True))
#TODO: Configure hyperparameters
EPOCHS = 100
learning_rate = 0.001

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


for batch, attn_mask in dataloader:
    init_logger(net, (batch.to(device), attn_mask.to(device)), dir=EXPERIMENT_DIRECTORY+"/tensorboard")
    break
for epoch in trange(EPOCHS):
    if epoch < RESUME:
        continue

    print(f"Beginning epoch {epoch}")

    last_batch = None
    last_generated = None
    running_total = 0
    num_runs = 0



    #TODO: Check training loop
    for batch, attn_mask in tqdm(dataloader,dynamic_ncols=True):
        optimizer.zero_grad()

        batch = batch.to(device)

        attn_mask = attn_mask.to(device)

        labels = batch[:, 1:].contiguous()
        batch = batch[:, :-1].contiguous()

        results = net(batch, padding_mask=attn_mask[:, :-1])

        loss = criterion(results.view(-1, results.size(-1)), labels.view(-1))

        loss.backward()

        #print(loss.item())

        running_total += loss.item()
        
        num_runs += 1


        optimizer.step()
        last_batch = batch[0].detach().cpu()
        last_generated = results[0].detach().cpu()

        
    
    num_test_runs = 1
    running_total_test = 0
    # with torch.no_grad():
    #     for batch, labels in tqdm(testloader):

    #         batch = batch.to(device)

    #         attn_mask = attn_mask.to(device)

    #         labels = batch[:, 1:].contiguous()
    #         batch = batch[:, :-1].contiguous()

    #         results = net(batch, padding_mask=attn_mask[:, :-1])

    #         loss = criterion(results.view(-1, results.size(-1)), labels.view(-1))


    #         running_total_test += loss.item()
            
    #         num_test_runs += 1



    print(f"Epoch {epoch}, Loss (Train): {running_total/num_runs}, Loss (Test): {running_total_test/num_test_runs}")


    log_data({
        "Loss/Train":running_total/num_runs,
        "Loss/Test":running_total_test/num_test_runs
        },epoch)
    with open(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt", "wb+") as f:
        torch.save(net.state_dict(),f)

    if epoch % 10 == 0 :
        with open(f"{EXPERIMENT_DIRECTORY}/ckpt/epoch_{epoch}.pt", "wb+") as f:
            torch.save(net.state_dict(),f)