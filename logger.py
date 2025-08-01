from torch.utils.tensorboard import SummaryWriter
import os


writer = None


def flush():
    global writer
    writer.flush()
    writer = None


def log_data(data, i):
    for key in data.keys():
        writer.add_scalar(key, data[key], i)


def log_img(img, name):
    writer.add_image(name, img)


def init_logger(net, data=None, dir="runs"):
    net.eval()
    global writer
    if not writer or writer is None:
        writer = SummaryWriter(dir)
    if data is not None:
        existing_files = [
            f for f in os.listdir(dir) if f.startswith("events.out.tfevents.")
        ]
        if not existing_files:
            writer.add_graph(net, data)
    # writer.close()
    net.train()
    os.system("tensorboard --logdir runs > /dev/null 2>&1 &")
    # os.system("sleep 5; open -a /Applications/Safari.app http://localhost:6006 &")
