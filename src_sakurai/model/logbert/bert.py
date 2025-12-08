import torch.nn as nn

from .encoder_block import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_len=512,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        is_logkey=True,
        is_time=False,
    ):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            max_len=max_len,
            is_logkey=is_logkey,
            is_time=is_time,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 2, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, segment_info=None, time_info=None):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info, time_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
