from architecture import Transformer
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import log_data, init_logger, log_img
import torchvision

import os
os.system(f"caffeinate -is -w {os.getpid()} &")



EXPERIMENT_DIRECTORY = "runs/tester"

os.mkdir(EXPERIMENT_DIRECTORY)

os.mkdir(EXPERIMENT_DIRECTORY+"/ckpt")



device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset())

testloader = get_dataloader(get_test_dataset())


net = Transformer()
net.to(device)
#TODO: Configure hyperparameters
EPOCHS = 500
learning_rate = 0.001

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



init_logger(net, next(iter(dataloader))[0].to(device), dir=EXPERIMENT_DIRECTORY+"/tensorboard")
for epoch in trange(EPOCHS):
    last_batch = None
    last_generated = None
    running_total = 0
    num_runs = 0



    #TODO: Check training loop
    for batch, attn_mask in tqdm(dataloader):
        optimizer.zero_grad()

        batch = batch.to(device)

        attn_mask = attn_mask.to(device)

        labels = batch[:, 1:].contiguous()
        batch = batch[:, :-1].contiguous()

        results = net(batch, padding_mask=attn_mask[:, :-1])

        loss = criterion(results.view(-1, results.size(-1)), labels.view(-1))

        loss.backward()

        running_total += loss.item()
        
        num_runs += 1


        optimizer.step()
        last_batch = batch[0].detach().cpu()
        last_generated = results[0].detach().cpu()
    
    num_test_runs = 0
    running_total_test = 0
    with torch.no_grad():
        for batch, labels in tqdm(testloader):

            batch = batch.to(device)

            attn_mask = attn_mask.to(device)

            labels = batch[:, 1:].contiguous()
            batch = batch[:, :-1].contiguous()

            results = net(batch, padding_mask=attn_mask[:, :-1])

            loss = criterion(results.view(-1, results.size(-1)), labels.view(-1))


            running_total_test += loss.item()
            
            num_test_runs += 1



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