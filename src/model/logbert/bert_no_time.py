import torch.nn as nn
import torch

from .encoder_block import TransformerBlock
from .embedding.bert_no_time import BERTEmbeddingNoTime


class BERTNoTime(nn.Module):
    """
    BERT without TimeEmbedding - TimeEmbeddingを完全に除去した軽量版
    
    元のBERTと同じアーキテクチャだが、TimeEmbedding関連のコードを含まない。
    is_logkeyとis_timeパラメータも不要。
    """

    def __init__(
        self,
        vocab_size,
        max_len=512,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
    ):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.embedding = BERTEmbeddingNoTime(
            vocab_size=vocab_size,
            embed_size=hidden,
            max_len=max_len,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 2, dropout)
                for _ in range(n_layers)
            ]
        )
        
        # 重み初期化を適用
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming初期化を適用してNaN問題を防止"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, segment_info=None, time_info=None):
        """
        time_info引数は互換性のために残しているが、使用しない
        """
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
