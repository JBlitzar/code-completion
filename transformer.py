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
    
class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim != None else dim
        
        self.block = nn.Sequential(
            nn.Linear(self.dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.dim),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.block(x)


#todo layernorm, mha, fact-check
class EncoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sa = SelfAttention()

        self.block = FeedForward()


    def forward(self, x):

        x = self.sa(x)

        x = self.block(x)

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sa = SelfAttention()

        self.eda = EncoderDecoderAttention()

        self.block = FeedForward()


    def forward(self, x, encoded):

        x = self.sa(x)

        x = self.eda(x, encoded)

        x = self.block(x)

        return x

#todo positional encoding, embedding I think, and figure out how the training loop/inference loop actually works
class Transformer(nn.Module):
    def __init__(self, num_blocks=6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks

        self.encoders = nn.ModuleList([EncoderBlock() for _ in range(num_blocks)])
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])

    # yoinked from JBlitzar/Diffusion
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
        10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x):
        for eidx, eblock in enumerate(self.encoders):
            x = eblock(x)

        encoded = x

        for didx, dblock in enumerate(self.decoders):
            x = dblock(x, encoded)
        
        return x

