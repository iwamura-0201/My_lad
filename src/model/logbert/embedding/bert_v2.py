import torch.nn as nn
from .token import TokenEmbedding
from .position_learnable import LearnablePositionalEmbedding
from .segment import SegmentEmbedding
from .time_embed import TimeEmbedding


class BERTv2Embedding(nn.Module):
    """
    BERTv2 Embedding - 学習可能な位置埋め込みを使用した改良版
    
    オリジナルBERTEmbeddingとの違い:
    - 固定sin/cosではなく学習可能な位置埋め込みを使用
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        max_len,
        dropout=0.1,
        is_logkey=True,
        is_time=False,
    ):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = LearnablePositionalEmbedding(
            d_model=self.token.embedding_dim, max_len=max_len
        )
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.time_embed = TimeEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.is_logkey = is_logkey
        self.is_time = is_time

    def forward(self, sequence, segment_label=None, time_info=None):
        x = self.position(sequence)
        x = x + self.token(sequence)
        if segment_label is not None:
            x = x + self.segment(segment_label)
        if self.is_time:
            x = x + self.time_embed(time_info)
        return self.dropout(x)
