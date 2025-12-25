"""
事前計算済みEventID埋め込みを使用するBERTモデル

BERTNoTimeをベースに、TokenEmbeddingの代わりに
PretrainedEventEmbeddingを使用する。
"""

import torch.nn as nn
import torch

from .encoder_block import TransformerBlock
from .embedding.bert_pretrained_embedding import BERTPretrainedEmbedding


class BERTWithPretrainedEmbedding(nn.Module):
    """
    事前計算済みEventID埋め込みを使用するBERTモデル
    
    BERTNoTimeと同様のアーキテクチャだが、TokenEmbeddingの代わりに
    事前計算済みのSentence-BERT埋め込みを使用する。
    
    主な特徴:
        - EventIDの埋め込みは事前計算済み（学習不要）
        - 特殊トークン（PAD, UNK, SOS, MASK）は学習可能
        - TimeEmbeddingは含まない（軽量版）
    """
    
    def __init__(
        self,
        pretrained_embed_path: str,
        eventid_mapping_path: str,
        max_len: int = 512,
        hidden: int = 384,
        n_layers: int = 12,
        attn_heads: int = 12,
        dropout: float = 0.1,
        freeze_pretrained: bool = True,
    ):
        """
        Args:
            pretrained_embed_path: 事前計算済み埋め込みのnpyファイルパス
            eventid_mapping_path: EventID→インデックスマッピングのJSONファイルパス
            max_len: 最大シーケンス長
            hidden: 隠れ層の次元
            n_layers: Transformerレイヤー数
            attn_heads: Attentionヘッド数
            dropout: ドロップアウト率
            freeze_pretrained: 事前計算済み埋め込みを凍結するかどうか
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        
        # 事前計算済み埋め込みを使用するBERT埋め込み層
        self.embedding = BERTPretrainedEmbedding(
            pretrained_embed_path=pretrained_embed_path,
            eventid_mapping_path=eventid_mapping_path,
            embed_size=hidden,
            max_len=max_len,
            dropout=dropout,
            freeze_pretrained=freeze_pretrained,
        )
        
        # Transformerブロック
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, hidden * 2, dropout)
            for _ in range(n_layers)
        ])
        
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
        Args:
            x: 入力シーケンス (batch_size, seq_len)
            segment_info: セグメント情報（オプション）
            time_info: 時間情報（互換性のために残すが使用しない）
        
        Returns:
            出力テンソル (batch_size, seq_len, hidden)
        """
        # Attentionマスクの作成（PADトークン=0を除外）
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        # 埋め込み
        x = self.embedding(x, segment_info)
        
        # Transformerブロックを通過
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x
    
    @property
    def vocab_size(self) -> int:
        """総ボキャブラリサイズ"""
        return self.embedding.vocab_size
