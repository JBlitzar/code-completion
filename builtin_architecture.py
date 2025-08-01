import torch.nn as nn
import torch
import math
import torch.nn.functional as F

# Shamelessly ripped from https://github.com/pytorch/examples/blob/main/word_language_model/model.py


class BuiltinPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(BuiltinPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class BuiltinTransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, embedding_drop=0.1
    ):
        super(BuiltinTransformerModel, self).__init__(
            d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers
        )
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = BuiltinPositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.embedding_dropout = nn.Dropout(embedding_drop)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        # archive-misc/test_new_attnmask.py
        return torch.triu(
            torch.full((sz, sz), float("-inf")), diagonal=1
        )  # torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True, transpose=True):
        # pov when goofy errors
        # maybe fixes?
        if transpose:
            src = src.transpose(0, 1)

        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.embedding_dropout(src)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)

        return F.log_softmax(output, dim=-1)


# def make_model():
#     vocab_size = 60
#     embed_dim = 128
#     heads = 2
#     ff_dim = 128
#     layers = 2
#     drop = 0


#     xformer_real = BuiltinTransformerModel(
#         vocab_size, embed_dim, heads, ff_dim, layers, drop
#     )  # nn.Transformer(d_model=128, nhead=1, num_decoder_layers=2, num_encoder_layers=0)
#     return xformer_real


def make_model():
    # an extra one just for luck
    vocab_size = 56730  # 22812#153128#3646#153128#5001
    embed_dim = 256
    heads = 4
    ff_dim = 256
    layers = 4
    drop = 0.1
    embedding_drop = 0.1

    xformer_real = BuiltinTransformerModel(
        vocab_size, embed_dim, heads, ff_dim, layers, drop, embedding_drop
    )  # nn.Transformer(d_model=128, nhead=1, num_decoder_layers=2, num_encoder_layers=0)
    return xformer_real


def make_model_custom(dim=256, heads=4, layers=4, drop=0.1, *args):
    # an extra one just for luck
    vocab_size = 22812  # 153128#3646#153128#5001
    embed_dim = dim
    heads = heads
    ff_dim = dim
    layers = layers
    drop = 0.1
    embedding_drop = 0.1

    xformer_real = BuiltinTransformerModel(
        vocab_size, embed_dim, heads, ff_dim, layers, drop, embedding_drop
    )  # nn.Transformer(d_model=128, nhead=1, num_decoder_layers=2, num_encoder_layers=0)
    return xformer_real


if __name__ == "__main__":
    model = make_model()
    print(model)
    print("Model created successfully.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
