import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbeddingNoTime(nn.Module):
    """
    BERT Embedding without TimeEmbedding - 軽量版
    
    TimeEmbedding関連のコードを完全に除去したバージョン。
    元のBERTEmbeddingは is_time=False でTimeEmbeddingを無効化できるが、
    このクラスではTimeEmbedding自体を含まないため、メモリ効率が良い。
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        max_len,
        dropout=0.1,
    ):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(
            d_model=self.token.embedding_dim, max_len=max_len
        )
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label=None, time_info=None):
        """
        time_info引数は互換性のために残しているが、使用しない
        """
        x = self.position(sequence)
        x = x + self.token(sequence)
        if segment_label is not None:
            x = x + self.segment(segment_label)
        return self.dropout(x)
