import torch.nn as nn
import torch


class LearnablePositionalEmbedding(nn.Module):
    """
    学習可能な位置埋め込み層。
    固定のsin/cosエンコーディングとは異なり、位置パターンをデータから学習する。
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # 位置インデックスを事前に生成
        position_ids = torch.arange(max_len).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

    def forward(self, x):
        """
        Args:
            x: 入力テンソル (batch_size, seq_len, ...) または (batch_size, seq_len)
        Returns:
            位置埋め込み (1, seq_len, d_model)
        """
        seq_len = x.size(1)
        position_ids = self.position_ids[:, :seq_len]
        return self.position_embeddings(position_ids)
