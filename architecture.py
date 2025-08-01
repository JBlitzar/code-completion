import torch
import torch.nn as nn
import numpy as np

DIM = 128
print(f"DIM IS SET TO {DIM}")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class MHA_SelfAttention(nn.Module):
    def __init__(self, embed_dim=DIM, num_heads=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if num_heads != 8:
            print(
                "Num heads is not 8. This is a reminder to change this back after experimenting with smaller architectures"
            )
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_heads = num_heads

    def forward(self, x, mask=None, triangle_mask=False):
        # if torch.isnan(x).any():
        #     print("NAN ALERT!")
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
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1).float()
            attn_mask = attn_mask.masked_fill(
                ~attn_mask.bool(), -1e9
            )  # https://github.com/pytorch/pytorch/issues/21518 we don't talk about how long that took to know. Later it seems like they also support bool, but idk 🤷

        # print(f"attn_mask shape: {attn_mask.shape if attn_mask is not None else None}")
        # if attn_mask is not None:
        #     print(f"attn_mask stats: max={attn_mask.max()}, min={attn_mask.min()}, mean={attn_mask.mean()}")

        x = x.transpose(0, 1)
        # if torch.isnan(x).any():
        #     print("NAN ALERT!")
        attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        attn_output = attn_output.transpose(0, 1)
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("NAN ALERT!")

        # if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
        #     print("NAN or INF detected in attn_output!")
        # print(f"Output stats: max={attn_output.max()}, min={attn_output.min()}, mean={attn_output.mean()}")

        return attn_output


class FeedForward(nn.Module):
    def __init__(self, dim=DIM, hidden_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim

        self.block = nn.Sequential(
            nn.LayerNorm(self.dim),  # nobody knows what this does
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sa = MHA_SelfAttention()
        self.block = FeedForward()

        # self.drop = nn.Dropout(p=0.1)

    def forward(self, x, padding_mask=None):
        res_x = x
        x = self.sa(x, mask=padding_mask, triangle_mask=True)

        # x = self.drop(x)
        x = x + res_x

        res_x_2 = x
        x = self.block(x)
        # x = self.drop(x)
        x = x + res_x_2

        # if torch.isnan(x).any():
        #     print("NAN ALERT!")

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, DIM, 2) * -(np.log(10000.0) / DIM))
        pe = torch.zeros(max_len, DIM)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


class DecoderTransformer(nn.Module):
    def __init__(self, num_blocks=6, vocab_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if vocab_size == 100:
            print(
                "WARNING: vocab_size is set to 100. You probably mean to set it to something else. Comment out the exit line below if this was intentional"
            )
            exit()

        self.num_blocks = num_blocks
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(num_blocks)])
        self.pos_encoding = PositionalEncoding()
        self.enc_embedding = nn.Embedding(vocab_size, DIM)

        self.oblock = nn.Sequential(
            nn.Linear(DIM, vocab_size),
            # nn.Softmax(dim=-1)
        )

        # https://github.com/hyunwoongko/transformer
        @torch.no_grad()
        def _initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.kaiming_uniform_(m.weight.data)

        self.apply(_initialize_weights)

        print(
            f"Model initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} params."
        )

    def forward(self, x, padding_mask=None):
        # if torch.isnan(x).any():
        #     print("NAN ALERT!")
        if isinstance(x, tuple):
            x, padding_mask = x

        if padding_mask is not None:
            padding_mask = padding_mask == 0

        x = self.pos_encoding(self.enc_embedding(x))

        # if torch.isnan(x).any():
        #     print("NAN ALERT!")

        for didx, dblock in enumerate(self.decoders):
            x = dblock(x, padding_mask=padding_mask)

        x = self.oblock(x)

        return x
