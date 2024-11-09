import torch
from torch.utils.tensorboard import SummaryWriter


writer = None
def log_data(data, i):

    
    for key in data.keys():
        writer.add_scalar(key, data[key], i)

def log_img(img, name):
    writer.add_image(name, img)



def init_logger(net, imgs, dir="runs"):
    net.eval()
    global writer
    if not writer:
        writer = SummaryWriter(dir)
    writer.add_graph(net, imgs)
    writer.close()
    net.train()