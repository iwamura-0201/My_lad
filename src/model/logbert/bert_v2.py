import torch.nn as nn
import torch

from .encoder_block import TransformerBlock
from .embedding.bert_v2 import BERTv2Embedding


class BERTv2(nn.Module):
    """
    BERTv2 - 改良版BERTモデル
    
    オリジナルBERTとの違い:
    1. 学習可能な位置埋め込み (LearnablePositionalEmbedding)
    2. 拡張されたFeed Forward層 (hidden * 4, オリジナルBERT準拠)
    """

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

        self.embedding = BERTv2Embedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            max_len=max_len,
            is_logkey=is_logkey,
            is_time=is_time,
        )

        # Feed Forward層を hidden * 4 に拡張 (オリジナルBERT準拠)
        # 元々は hidden * 2 だった
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(n_layers)
            ]
        )
        
        # 重み初期化を適用
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming初期化を適用してNaN問題を防止"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier normal初期化（Transformerに適している）
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 埋め込み層は小さな値で初期化
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, segment_info=None, time_info=None):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info, time_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
