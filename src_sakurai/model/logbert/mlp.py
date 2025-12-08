import torch.nn as nn

from .embedding import BERTEmbedding
from .encoder_block import MLPBlock
from einops import repeat


class MLP(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_len=512,
        hidden=768,
        n_layers=12,
        dropout=0.1,
        is_logkey=True,
        is_time=False,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers

        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            max_len=max_len,
            is_logkey=is_logkey,
            is_time=is_time,
        )
        self.mlp_blocks = nn.ModuleList(
            [MLPBlock(hidden, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info=None, time_info=None):
        mask = x > 0
        mask = repeat(mask, "b t -> b t 1")

        x = self.embedding(x, segment_info, time_info)
        for mlp in self.mlp_blocks:
            x = mlp.forward(x, mask)

        return x
