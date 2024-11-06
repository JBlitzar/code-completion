import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x):
        # do stuff
        q = None
        k = None
        v = None

        dk = None

        z = F.softmax((q @ k) / np.sqrt(dk)) @ v

        return z

class EncoderDecoderAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x, encoded):
        return x

class EncoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sa = SelfAttention()

        self.block = nn.Sequential()


    def forward(self, x):

        x = self.sa(x)

        x = self.block(x)

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sa = SelfAttention()

        self.eda = EncoderDecoderAttention()

        self.block = nn.Sequential()


    def forward(self, x, encoded):

        x = self.sa(x)

        x = self.eda(x, encoded)

        x = self.block(x)

        return x


class Transformer(nn.Module):
    def __init__(self, num_blocks=6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks

        self.encoders = []
        self.decoders = []

        for _ in range(num_blocks):
            self.encoders.append(EncoderBlock())
            self.decoders.append(DecoderBlock())
    

    def forward(self, x):
        for eidx, eblock in enumerate(self.encoders):
            x = eblock(x)

        encoded = x

        for didx, dblock in enumerate(self.decoders):
            x = dblock(x, encoded)
        
        return x

