import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers?tab=readme-ov-file#queries-keys-and-values
class SelfAttention(nn.Module):
    def __init__(self, embed_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        
        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)

        self.sqrt_dk = np.sqrt(self.embed_dim)


    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)


        z = F.softmax(
            (
                # transpose last 2 dimensions
                q @ k.transpose(-2, -1)
            ) / self.sqrt_dk
        , dim=-1) @ v

        return z

class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        
        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)

        self.sqrt_dk = np.sqrt(self.embed_dim)


    def forward(self, x, encoded):
        # Literally the same thing except K and V are encoded
        q = self.query(x)
        k = self.key(encoded)
        v = self.value(encoded)


        z = F.softmax(
            (
            
                q @ k.transpose(-2, -1)
            ) / self.sqrt_dk
        , dim=-1) @ v

        return z

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

        self.encoders = nn.ModuleList([EncoderBlock() for _ in range(num_blocks)])
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])
    

    def forward(self, x):
        for eidx, eblock in enumerate(self.encoders):
            x = eblock(x)

        encoded = x

        for didx, dblock in enumerate(self.decoders):
            x = dblock(x, encoded)
        
        return x

