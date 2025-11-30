import torch.nn as nn

from .embedding import BERTEmbedding
from .encoder_block import LSTMBlock, GRUBlock
from einops import repeat


class LSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_len=512,
        hidden=768,
        n_layers=12,
        dropout=0.1,
        bidirectional=True,
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
        self.lstm_blocks = nn.ModuleList(
            [LSTMBlock(hidden, dropout, bidirectional) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info=None, time_info=None):
        mask = x > 0
        mask = repeat(mask, "b t -> b t 1")
        x = self.embedding(x, segment_info, time_info)
        for lstm in self.lstm_blocks:
            x = lstm.forward(x, mask)
        return x


class GRU(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_len=512,
        hidden=768,
        n_layers=12,
        dropout=0.1,
        bidirectional=True,
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
        self.gru_blocks = nn.ModuleList(
            [GRUBlock(hidden, dropout, bidirectional) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info=None, time_info=None):
        mask = x > 0
        mask = repeat(mask, "b t -> b t 1")
        x = self.embedding(x, segment_info, time_info)
        for gru in self.gru_blocks:
            x = gru.forward(x, mask)

        return x
