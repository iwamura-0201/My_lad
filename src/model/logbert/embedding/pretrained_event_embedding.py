"""
事前計算済みEventID埋め込みを使用する埋め込みモジュール

Sentence-BERTで生成された埋め込みをEventIDのルックアップテーブルとして使用する。
特殊トークン（PAD, UNK, SOS, MASK）は学習可能なパラメータとして追加。
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class PretrainedEventEmbedding(nn.Module):
    """
    事前計算済みのSentence-BERT埋め込みを使用するEventID埋め込み層
    
    特殊トークン（PAD, UNK, SOS, MASK）は学習可能なパラメータとして追加。
    事前計算済み埋め込みは固定（学習しない）。
    
    インデックスマッピング:
        0: PAD（パディング）
        1: UNK（未知トークン）
        2: SOS（文開始）
        3: MASK（マスク）
        4+: EventID埋め込み
    """
    
    # 特殊トークンの数
    NUM_SPECIAL_TOKENS = 4
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    MASK_IDX = 3
    
    def __init__(
        self,
        pretrained_embed_path: str,
        eventid_mapping_path: str,
        output_dim: int = None,
        freeze_pretrained: bool = True,
    ):
        """
        Args:
            pretrained_embed_path: 事前計算済み埋め込みのnpyファイルパス
            eventid_mapping_path: EventID→インデックスマッピングのJSONファイルパス
            output_dim: 出力次元（Noneの場合は埋め込み次元をそのまま使用）
            freeze_pretrained: 事前計算済み埋め込みを凍結するかどうか
        """
        super().__init__()
        
        # 埋め込みデータの読み込み
        pretrained_embeds = np.load(pretrained_embed_path)
        self.pretrained_dim = pretrained_embeds.shape[1]
        self.num_events = pretrained_embeds.shape[0]
        
        # EventID → インデックスマッピングの読み込み
        with open(eventid_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        # EventID文字列 → 埋め込みインデックス（特殊トークン分をオフセット）
        self.eventid_to_idx = {
            str(eid): idx + self.NUM_SPECIAL_TOKENS 
            for idx, eid in enumerate(mapping_data['event_ids'])
        }
        
        # 特殊トークン用の学習可能埋め込み
        self.special_embeds = nn.Embedding(
            num_embeddings=self.NUM_SPECIAL_TOKENS,
            embedding_dim=self.pretrained_dim,
            padding_idx=self.PAD_IDX,
        )
        # 特殊トークン埋め込みの初期化
        nn.init.normal_(self.special_embeds.weight, mean=0.0, std=0.02)
        # PADトークンは0ベクトルに固定
        self.special_embeds.weight.data[self.PAD_IDX].zero_()
        
        # 事前計算済み埋め込みの登録
        if freeze_pretrained:
            # 学習しない場合はbufferとして登録
            self.register_buffer(
                'pretrained_embeds',
                torch.tensor(pretrained_embeds, dtype=torch.float32)
            )
        else:
            # 学習する場合はParameterとして登録
            self.pretrained_embeds = nn.Parameter(
                torch.tensor(pretrained_embeds, dtype=torch.float32)
            )
        
        # 出力次元への投影層（必要な場合）
        self.output_dim = output_dim if output_dim else self.pretrained_dim
        if output_dim and output_dim != self.pretrained_dim:
            self.projection = nn.Linear(self.pretrained_dim, output_dim)
        else:
            self.projection = None
        
        # 埋め込み次元（外部からのアクセス用）
        self.embedding_dim = self.output_dim
    
    def get_event_index(self, event_id: str) -> int:
        """EventID文字列から埋め込みインデックスを取得"""
        return self.eventid_to_idx.get(str(event_id), self.UNK_IDX)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: インデックステンソル (batch_size, seq_len)
               0-3: 特殊トークン
               4+: EventIDインデックス
        
        Returns:
            埋め込みテンソル (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = x.shape
        
        # 特殊トークンかどうかのマスク
        is_special = x < self.NUM_SPECIAL_TOKENS
        
        # 出力テンソルの初期化
        output = torch.zeros(
            batch_size, seq_len, self.pretrained_dim,
            device=x.device, dtype=self.pretrained_embeds.dtype
        )
        
        # 特殊トークンの埋め込み
        special_mask = is_special
        if special_mask.any():
            special_indices = x[special_mask]
            output[special_mask] = self.special_embeds(special_indices)
        
        # 事前計算済み埋め込み
        event_mask = ~is_special
        if event_mask.any():
            # EventIDインデックスを事前計算済み埋め込みのインデックスに変換
            event_indices = x[event_mask] - self.NUM_SPECIAL_TOKENS
            # インデックス範囲を制限（範囲外はUNKとして扱う）
            event_indices = event_indices.clamp(0, self.num_events - 1)
            output[event_mask] = self.pretrained_embeds[event_indices]
        
        # 投影層（必要な場合）
        if self.projection is not None:
            output = self.projection(output)
        
        return output
    
    @property
    def vocab_size(self) -> int:
        """総ボキャブラリサイズ（特殊トークン + EventID数）"""
        return self.NUM_SPECIAL_TOKENS + self.num_events
