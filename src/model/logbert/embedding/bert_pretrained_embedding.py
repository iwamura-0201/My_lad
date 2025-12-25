"""
事前計算済みEventID埋め込みを使用するBERT埋め込み層

PretrainedEventEmbeddingを使用してTokenEmbeddingを置き換える。
"""

import torch.nn as nn
from .pretrained_event_embedding import PretrainedEventEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTPretrainedEmbedding(nn.Module):
    """
    事前計算済みEventID埋め込みを使用するBERT埋め込み層
    
    BERTEmbeddingNoTimeをベースに、TokenEmbeddingの代わりに
    PretrainedEventEmbeddingを使用する。
    
    構成:
        1. PretrainedEventEmbedding: 事前計算済みEventID埋め込み
        2. PositionalEmbedding: 位置情報（sin/cos）
        3. SegmentEmbedding: セグメント情報（オプション）
    """
    
    def __init__(
        self,
        pretrained_embed_path: str,
        eventid_mapping_path: str,
        embed_size: int,
        max_len: int,
        dropout: float = 0.1,
        freeze_pretrained: bool = True,
    ):
        """
        Args:
            pretrained_embed_path: 事前計算済み埋め込みのnpyファイルパス
            eventid_mapping_path: EventID→インデックスマッピングのJSONファイルパス
            embed_size: 出力埋め込み次元
            max_len: 最大シーケンス長
            dropout: ドロップアウト率
            freeze_pretrained: 事前計算済み埋め込みを凍結するかどうか
        """
        super().__init__()
        
        # 事前計算済みEventID埋め込み
        self.event_embed = PretrainedEventEmbedding(
            pretrained_embed_path=pretrained_embed_path,
            eventid_mapping_path=eventid_mapping_path,
            output_dim=embed_size,
            freeze_pretrained=freeze_pretrained,
        )
        
        # 位置埋め込み（sin/cos）
        self.position = PositionalEmbedding(
            d_model=embed_size,
            max_len=max_len,
        )
        
        # セグメント埋め込み
        self.segment = SegmentEmbedding(embed_size=embed_size)
        
        # ドロップアウト
        self.dropout = nn.Dropout(p=dropout)
        
        # 出力次元
        self.embed_size = embed_size
    
    def forward(self, sequence, segment_label=None, time_info=None):
        """
        Args:
            sequence: 入力シーケンス (batch_size, seq_len)
            segment_label: セグメントラベル（オプション）
            time_info: 時間情報（互換性のために残すが使用しない）
        
        Returns:
            埋め込みテンソル (batch_size, seq_len, embed_size)
        """
        # EventID埋め込み
        x = self.event_embed(sequence)
        
        # 位置埋め込みを加算
        x = x + self.position(sequence)
        
        # セグメント埋め込みを加算（指定された場合）
        if segment_label is not None:
            x = x + self.segment(segment_label)
        
        return self.dropout(x)
    
    @property
    def vocab_size(self) -> int:
        """総ボキャブラリサイズ"""
        return self.event_embed.vocab_size
