import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DIM = 512

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#TODO: check that all these things work how I think they will


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
        self.num_heads = num_heads

    def forward(self, x, mask=None, triangle_mask=False):
        attn_mask = None
        seq_len = x.size(1)
        
        if triangle_mask:

            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
            attn_mask = attn_mask.to(x.device)
        if mask is not None:
            if attn_mask is not None:
                attn_mask = mask.unsqueeze(1) & attn_mask.unsqueeze(0)
            else:
                attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
        

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        
        x = x.transpose(0, 1)
        attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        attn_output = attn_output.transpose(0, 1)
        
        return attn_output

class MHA_EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_dim=DIM, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_heads = num_heads

    def forward(self, x, encoded, mask=None):
        attn_mask = None
        seq_len_x = x.size(1)
        seq_len_encoded = encoded.size(1)


        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, seq_len_x, seq_len_encoded)

            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        x = x.transpose(0, 1)
        encoded = encoded.transpose(0, 1)

        attn_output, _ = self.mha(x, encoded, encoded, attn_mask=attn_mask)
        
        attn_output = attn_output.transpose(0, 1)
        
        return attn_output


    
class FeedForward(nn.Module):
    def __init__(self, dim=DIM, hidden_dim = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim != None else dim
        
        self.block = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim,self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim,self.dim),
            nn.GELU(),
           

        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sa = MHA_SelfAttention()

        self.block = FeedForward()


    def forward(self, x, padding_mask=None):

        res_x = x.clone()

        x = self.sa(x, padding_mask)

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


    def forward(self, x, encoded, padding_mask=None):

        res_x = x.clone()


        x = self.sa(x, mask=padding_mask, triangle_mask=True)

        x = x + res_x

        res_x_2 = x.clone()

        x = self.eda(x, encoded, mask=padding_mask)

        x = x + res_x_2

        res_x_3 = x.clone()

        x = self.block(x)

        x = x + res_x_3

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, DIM, 2) * -(np.log(10000.0) / DIM))
        pe = torch.zeros(max_len, DIM)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        #print(f"x shape: {x.shape}, pe shape: {self.pe[:, :seq_len, :].shape}")
        return x + self.pe[:, :seq_len, :].to(x.device)
    
#todo figure out how the training loop/inference loop actually works
class Transformer(nn.Module):
    def __init__(self, num_blocks=6, vocab_size=100,seq_len=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks

        self.encoders = nn.ModuleList([EncoderBlock() for _ in range(num_blocks)])
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])

        self.pos_encoding = PositionalEncoding()

        self.enc_embedding = nn.Embedding(vocab_size,DIM)


        self.oblock = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.Softmax(dim=-1)
        )



    def forward(self, x, padding_mask=None):

        if isinstance(x, tuple):
            x, padding_mask = x

        
        if(padding_mask is not None):
            padding_mask = padding_mask == 0
        
        x = self.pos_encoding(self.enc_embedding(x))

        #print(f"After embedding and pos encoding, x shape: {x.shape}")


        for eidx, eblock in enumerate(self.encoders):
            x = eblock(x, padding_mask=padding_mask)


        #print(f"After encoder, x shape: {x.shape}")


        encoded = x.clone()


        x = self.pos_encoding(x)

        #print(f"After 2nd pos encoding, x shape: {x.shape}")

        for didx, dblock in enumerate(self.decoders):
            x = dblock(x, encoded, padding_mask=padding_mask)
        


        x = self.oblock(x)

        return x

