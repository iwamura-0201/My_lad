"""
ScoreBERT: BERT-Based Self-Supervised Model for Log Anomaly Detection

This module implements a bidirectional transformer model that predicts
masked token embeddings and scores using self-supervised learning.

Input: Each token is a vector of [EventID_emb (384-dim) + score (1) + score_missing (1)]
Output: Predicted score and EventID embedding for masked positions
"""

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.pe[:, :x.size(1)]


class ScoreBERTEmbedding(nn.Module):
    """
    ScoreBERT Embedding: Linear projection + Positional Embedding
    
    Input tokens are projected to hidden_size, then positional embeddings are added.
    """

    def __init__(self, input_dim=386, hidden_size=128, max_len=512, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.position = PositionalEmbedding(d_model=hidden_size, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - input tokens
        Returns:
            (batch, seq_len, hidden_size)
        """
        # Project input to hidden size
        x = self.input_projection(x)
        # Add positional embedding
        x = x + self.position(x)
        # Layer norm and dropout
        x = self.layer_norm(x)
        return self.dropout(x)


class Attention(nn.Module):
    """Scaled Dot-Product Attention."""

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalization."""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """Bidirectional Transformer Block."""

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class ScoreBERT(nn.Module):
    """
    ScoreBERT: A BERT-based model for self-supervised learning on log sequences.
    
    Architecture:
        1. Input Projection: Linear(input_dim -> hidden_size)
        2. Positional Embedding: Sinusoidal encoding
        3. Transformer Blocks: Bidirectional self-attention
        4. Prediction Heads: Score and EventID embedding prediction
    
    Args:
        input_dim: Dimension of input tokens (default: 386 = 384 EventID_emb + 1 score + 1 score_missing)
        hidden_size: Hidden dimension for transformer (128-256)
        n_layers: Number of transformer layers (2-4)
        n_heads: Number of attention heads (4-8)
        dropout: Dropout rate (default: 0.1)
        max_len: Maximum sequence length (default: 512)
        eventid_dim: Dimension of EventID embedding (default: 384)
    """

    def __init__(
        self,
        input_dim=386,
        hidden_size=128,
        n_layers=3,
        n_heads=4,
        dropout=0.1,
        max_len=512,
        eventid_dim=384,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.eventid_dim = eventid_dim

        # Embedding layer
        self.embedding = ScoreBERTEmbedding(
            input_dim=input_dim,
            hidden_size=hidden_size,
            max_len=max_len,
            dropout=dropout,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden=hidden_size,
                attn_heads=n_heads,
                feed_forward_hidden=hidden_size * 4,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Prediction heads
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        
        self.eventid_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, eventid_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier normal for transformers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        """
        Forward pass of ScoreBERT.
        
        Args:
            x: (batch, seq_len, input_dim) - input sequence with masked tokens
            mask: (batch, seq_len) - attention mask (optional)
            
        Returns:
            score_pred: (batch, seq_len, 1) - predicted scores
            eventid_pred: (batch, seq_len, eventid_dim) - predicted EventID embeddings
        """
        # Create attention mask if not provided
        if mask is not None:
            # Expand mask for attention: (batch, 1, seq_len, seq_len)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, -1, x.size(1), -1)
        else:
            attn_mask = None

        # Embedding
        x = self.embedding(x)

        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, attn_mask)

        # Predictions
        score_pred = self.score_head(x)
        eventid_pred = self.eventid_head(x)

        return score_pred, eventid_pred

    def get_masked_input(self, x, mask_positions, mask_value=0.0):
        """
        Create masked input by zeroing out specified positions.
        
        Args:
            x: (batch, seq_len, input_dim) - original input
            mask_positions: (batch, seq_len) - boolean mask indicating positions to mask
            mask_value: Value to use for masking (default: 0.0)
            
        Returns:
            masked_x: (batch, seq_len, input_dim) - input with masked positions
        """
        masked_x = x.clone()
        # Mask EventID_emb (first 384 dims) and score (dim 384) 
        # Keep score_missing (dim 385) intact
        mask_expanded = mask_positions.unsqueeze(-1).expand(-1, -1, self.eventid_dim + 1)
        masked_x[:, :, :self.eventid_dim + 1] = torch.where(
            mask_expanded,
            torch.full_like(masked_x[:, :, :self.eventid_dim + 1], mask_value),
            masked_x[:, :, :self.eventid_dim + 1]
        )
        return masked_x
