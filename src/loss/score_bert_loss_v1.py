"""
ScoreBERT Loss Function

Custom loss function for ScoreBERT with:
- Score MSE loss (excluding score_missing=1 positions)
- EventID embedding MSE loss
- Dynamic normalization using running statistics (μ, σ)
- Weighted combination: final = score_err_norm + β * event_err_norm
"""

import torch
import torch.nn as nn


class RunningStats:
    """
    Track running mean and standard deviation for loss normalization.
    Uses exponential moving average for stability.
    """

    def __init__(self, momentum=0.1, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.initialized = False

    def update(self, value):
        """Update running statistics with a new value."""
        if not self.initialized:
            self.running_mean = value.item() if torch.is_tensor(value) else value
            self.running_var = 1.0
            self.initialized = True
        else:
            val = value.item() if torch.is_tensor(value) else value
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * val
            # Update variance using Welford's online algorithm approximation
            diff = val - self.running_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (diff ** 2)

    def normalize(self, value):
        """Normalize a value using running statistics."""
        if not self.initialized:
            return value
        std = (self.running_var ** 0.5) + self.epsilon
        return (value - self.running_mean) / std

    @property
    def mean(self):
        return self.running_mean if self.initialized else 0.0

    @property
    def std(self):
        return (self.running_var ** 0.5) if self.initialized else 1.0


class ScoreBERTLoss(nn.Module):
    """
    Loss function for ScoreBERT self-supervised learning.
    
    Combines two losses with dynamic normalization:
    1. score_err: MSE on masked score positions (excluding score_missing=1)
    2. eventid_err: MSE on masked EventID embedding positions
    
    Final loss = score_err_norm + β * eventid_err_norm
    
    Where normalization is: err_norm = (err - μ) / (σ + ε)
    
    Args:
        beta: Weight for eventid_err (default: 1.0, typically sufficient after normalization)
        epsilon: Small value for numerical stability
        momentum: Momentum for running statistics update
    """

    def __init__(self, beta=1.0, epsilon=1e-8, momentum=0.1):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        
        # Running statistics for normalization
        self.score_stats = RunningStats(momentum=momentum, epsilon=epsilon)
        self.eventid_stats = RunningStats(momentum=momentum, epsilon=epsilon)

    def forward(
        self,
        pred_score,        # (batch, seq_len, 1)
        pred_eventid,      # (batch, seq_len, eventid_dim)
        target_score,      # (batch, seq_len, 1)
        target_eventid,    # (batch, seq_len, eventid_dim)
        mask_positions,    # (batch, seq_len) bool - positions that were masked
        score_missing,     # (batch, seq_len) - 1 if score is missing, 0 otherwise
    ):
        """
        Compute the combined normalized loss.
        
        Args:
            pred_score: Predicted scores from model
            pred_eventid: Predicted EventID embeddings from model
            target_score: Original (ground truth) scores
            target_eventid: Original (ground truth) EventID embeddings
            mask_positions: Boolean mask indicating which positions were masked
            score_missing: Indicator for missing scores (1 = missing, 0 = valid)
            
        Returns:
            loss: Combined normalized loss
            loss_dict: Dictionary with individual loss components for logging
        """
        batch_size = pred_score.size(0)
        
        # ============================================
        # 1. Score Error (exclude score_missing=1)
        # ============================================
        # Valid positions: masked AND score is not missing
        score_valid_mask = mask_positions & (score_missing == 0)
        
        if score_valid_mask.sum() > 0:
            # Select valid positions
            pred_score_flat = pred_score.squeeze(-1)  # (batch, seq_len)
            target_score_flat = target_score.squeeze(-1)  # (batch, seq_len)
            
            score_diff = (pred_score_flat - target_score_flat) ** 2
            score_err = (score_diff * score_valid_mask.float()).sum() / (score_valid_mask.sum() + self.epsilon)
        else:
            score_err = torch.tensor(0.0, device=pred_score.device)

        # ============================================
        # 2. EventID Embedding Error (DO NOT exclude score_missing=1)
        # ============================================
        if mask_positions.sum() > 0:
            # Expand mask for eventid dimension
            mask_expanded = mask_positions.unsqueeze(-1).expand_as(pred_eventid)
            
            eventid_diff = (pred_eventid - target_eventid) ** 2
            eventid_err = (eventid_diff * mask_expanded.float()).sum() / (mask_expanded.sum() + self.epsilon)
        else:
            eventid_err = torch.tensor(0.0, device=pred_eventid.device)

        # ============================================
        # 3. Dynamic Normalization
        # ============================================
        if self.training:
            # Update running statistics during training
            self.score_stats.update(score_err.detach())
            self.eventid_stats.update(eventid_err.detach())
        
        # Normalize errors
        score_err_norm = self.score_stats.normalize(score_err)
        eventid_err_norm = self.eventid_stats.normalize(eventid_err)

        # ============================================
        # 4. Combined Loss
        # ============================================
        final_loss = score_err_norm + self.beta * eventid_err_norm

        # Return loss and components for logging
        loss_dict = {
            'score_err': score_err.item(),
            'eventid_err': eventid_err.item(),
            'score_err_norm': score_err_norm.item() if torch.is_tensor(score_err_norm) else score_err_norm,
            'eventid_err_norm': eventid_err_norm.item() if torch.is_tensor(eventid_err_norm) else eventid_err_norm,
            'final_loss': final_loss.item() if torch.is_tensor(final_loss) else final_loss,
            'score_mean': self.score_stats.mean,
            'score_std': self.score_stats.std,
            'eventid_mean': self.eventid_stats.mean,
            'eventid_std': self.eventid_stats.std,
        }

        return final_loss, loss_dict

    def get_stats(self):
        """Return current running statistics."""
        return {
            'score_mean': self.score_stats.mean,
            'score_std': self.score_stats.std,
            'eventid_mean': self.eventid_stats.mean,
            'eventid_std': self.eventid_stats.std,
        }


class ScoreBERTLossSimple(nn.Module):
    """
    Simplified version without running normalization (for evaluation/inference).
    
    Uses fixed normalization or no normalization.
    """

    def __init__(self, beta=1.0, score_mean=0.0, score_std=1.0, 
                 eventid_mean=0.0, eventid_std=1.0, epsilon=1e-8):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.score_mean = score_mean
        self.score_std = score_std
        self.eventid_mean = eventid_mean
        self.eventid_std = eventid_std

    def forward(
        self,
        pred_score,
        pred_eventid,
        target_score,
        target_eventid,
        mask_positions,
        score_missing,
    ):
        # Score error
        score_valid_mask = mask_positions & (score_missing == 0)
        if score_valid_mask.sum() > 0:
            pred_score_flat = pred_score.squeeze(-1)
            target_score_flat = target_score.squeeze(-1)
            score_diff = (pred_score_flat - target_score_flat) ** 2
            score_err = (score_diff * score_valid_mask.float()).sum() / (score_valid_mask.sum() + self.epsilon)
        else:
            score_err = torch.tensor(0.0, device=pred_score.device)

        # EventID error
        if mask_positions.sum() > 0:
            mask_expanded = mask_positions.unsqueeze(-1).expand_as(pred_eventid)
            eventid_diff = (pred_eventid - target_eventid) ** 2
            eventid_err = (eventid_diff * mask_expanded.float()).sum() / (mask_expanded.sum() + self.epsilon)
        else:
            eventid_err = torch.tensor(0.0, device=pred_eventid.device)

        # Normalize with fixed stats
        score_err_norm = (score_err - self.score_mean) / (self.score_std + self.epsilon)
        eventid_err_norm = (eventid_err - self.eventid_mean) / (self.eventid_std + self.epsilon)

        final_loss = score_err_norm + self.beta * eventid_err_norm

        return final_loss, {
            'score_err': score_err.item(),
            'eventid_err': eventid_err.item(),
            'final_loss': final_loss.item(),
        }
