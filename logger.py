import torch
from torch.utils.tensorboard import SummaryWriter
import os
import webbrowser



writer = None
def log_data(data, i):

    
    for key in data.keys():
        writer.add_scalar(key, data[key], i)

def log_img(img, name):
    writer.add_image(name, img)



def init_logger(net, data=None, dir="runs"):
    net.eval()
    global writer
    if not writer:
        writer = SummaryWriter(dir)
    if data is not None:
        writer.add_graph(net, data)
    writer.close()
    net.train()
    os.system("tensorboard --logdir runs > /dev/null 2>&1 &")
    os.system("sleep 5; open http://localhost:6006 &")