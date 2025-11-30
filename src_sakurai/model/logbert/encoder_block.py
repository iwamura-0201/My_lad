import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class LSTMBlock(nn.Module):
    def __init__(self, hidden, dropout, bidirectional):
        super().__init__()
        if bidirectional:
            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=hidden // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask

        x, _ = self.lstm(x)

        return self.dropout(x)


class GRUBlock(nn.Module):
    def __init__(self, hidden, dropout, bidirectional):
        super().__init__()
        if bidirectional:
            self.gru = nn.GRU(
                input_size=hidden,
                hidden_size=hidden // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            self.gru = nn.GRU(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x, _ = self.gru(x)
        return self.dropout(x)


class MLPBlock(nn.Module):

    def __init__(self, hidden, dropout):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = self.linear(x)
        x = self.activation(x)
        return self.dropout(x)
