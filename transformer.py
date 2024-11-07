import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DIM = 512

#todo check that all these things work how I think they will


# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers?tab=readme-ov-file#queries-keys-and-values

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=DIM, *args, **kwargs):
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
    def __init__(self, embed_dim=DIM, *args, **kwargs):
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
    


class MHA_SelfAttention(nn.Module):
    def __init__(self, embed_dim=DIM, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.transpose(0, 1)
        
        attn_output, _ = self.mha(x, x, x)
        
        attn_output = attn_output.transpose(0, 1)
        
        return attn_output

class MHA_EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_dim=DIM, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, encoded):

        x = x.transpose(0, 1)
        encoded = encoded.transpose(0, 1)
        

        attn_output, _ = self.mha(x, encoded, encoded)
        

        attn_output = attn_output.transpose(0, 1)
        
        return attn_output

    
class FeedForward(nn.Module):
    def __init__(self, dim=DIM, hidden_dim = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim != None else dim
        
        self.block = nn.Sequential(
            nn.Linear(self.dim,self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim,self.dim),
            nn.GELU()

        )

    def forward(self, x):
        return self.block(x)


#todo layernorm, fact-check, residual apparently?
class EncoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sa = MHA_SelfAttention()

        self.block = FeedForward()


    def forward(self, x):

        res_x = x.clone()

        x = self.sa(x)

        x = x + res_x

        res_x_2 = x.clone()

        x = self.block(x)

        x = x + res_x_2

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sa = MHA_SelfAttention()

        self.eda = MHA_EncoderDecoderAttention()

        self.block = FeedForward()


    def forward(self, x, encoded):

        res_x = x.clone()

        # Allegedly needs to be masked?
        x = self.sa(x)

        x = x + res_x

        res_x_2 = x.clone()

        x = self.eda(x, encoded)

        x = x + res_x_2

        res_x_3 = x.clone()

        x = self.block(x)

        x = x + res_x_3

        return x

#todo figure out how the training loop/inference loop actually works
class Transformer(nn.Module):
    def __init__(self, num_blocks=6, vocab_size=100,seq_len=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks

        self.encoders = nn.ModuleList([EncoderBlock() for _ in range(num_blocks)])
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])

        self.e_lnorm = nn.LayerNorm(DIM)
        self.d_lnorm = nn.LayerNorm(DIM)

        self.pos_encoding = self.get_pos_encoding(torch.range(seq_len),DIM)

        self.enc_embedding = nn.Embedding(vocab_size,DIM)

        self.dec_embedding = nn.Embedding(vocab_size,DIM)

        self.ublock = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.Softmax()
        )


    # yoinked from JBlitzar/Diffusion
    def get_pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
        10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x):
        
        x = self.enc_embedding(x) + self.pos_encoding


        for eidx, eblock in enumerate(self.encoders):
            x = eblock(x)

        x = self.e_lnorm(x)


        encoded = x.clone()

        x = self.dec_embedding(x) + self.pos_encoding

        for didx, dblock in enumerate(self.decoders):
            x = dblock(x, encoded)
        
        x = self.d_lnorm(x)

        x = self.ublock(x)

        return x

