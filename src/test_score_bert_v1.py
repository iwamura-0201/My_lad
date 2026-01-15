"""
ScoreBERT Evaluation Script (V1)

Evaluate a trained ScoreBERT model on test data using configurable scoring.

Window-level anomaly score (configurable):
    score_part  = AGG_score(raw_score_err)    # default: max
    event_part  = AGG_event(event_context_err) # default: P95
    window_score = score_part + γ * event_part

Features:
- Configurable aggregation methods for score and event errors
- Load test_normal and test_abnormal data
- Inference without masking
- ROC-AUC and PR-AUC visualization
- P/R/F1 curves at various thresholds

Usage:
    python test_score_bert_v1.py <model_dir>
    python test_score_bert_v1.py <model_dir> --gamma 0.5 --score-agg max --event-agg pq
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.logbert.score_bert import ScoreBERT
from train_score_bert import ScoreBERTDataset


# =============================================================================
# Scoring Configuration (shared with diagnosis_v1.py)
# =============================================================================

AggregationType = Literal['max', 'mean', 'pq']


@dataclass
class ScoringConfig:
    """
    Centralized scoring configuration.
    
    Change the aggregation methods here to modify the scoring formula:
        window_score = score_part + gamma * event_part
    
    Where:
        score_part = AGG_score(raw_score_err)
        event_part = AGG_event(event_context_err)
    """
    score_agg: AggregationType = 'max'
    event_agg: AggregationType = 'pq'
    percentile: float = 95
    gamma: float = 1.0
    
    def get_score_label(self) -> str:
        if self.score_agg == 'pq':
            return f'P{int(self.percentile)}'
        return self.score_agg.upper()
    
    def get_event_label(self) -> str:
        if self.event_agg == 'pq':
            return f'P{int(self.percentile)}'
        return self.event_agg.upper()
    
    def get_formula_string(self) -> str:
        score_label = self.get_score_label()
        event_label = self.get_event_label()
        return f"window_score = {score_label}(raw_score_err) + {self.gamma} × {event_label}(event_context_err)"


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_config(model_dir):
    """Load trained model and config."""
    model_dir = Path(model_dir)
    
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    
    device = cfg.default.device_id
    if not torch.cuda.is_available() and 'cuda' in device:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    model = ScoreBERT(
        input_dim=cfg.network.encoder.input_dim,
        hidden_size=cfg.network.encoder.hidden_size,
        n_layers=cfg.network.encoder.n_layers,
        n_heads=cfg.network.encoder.n_heads,
        dropout=cfg.network.encoder.dropout,
        max_len=cfg.network.encoder.max_len,
        eventid_dim=cfg.network.encoder.eventid_dim,
    )
    
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_dir}")
    print(f"Best epoch: {checkpoint['epoch']}, Best loss: {checkpoint['loss']:.4f}")
    
    return model, cfg, device


# =============================================================================
# Score Computation
# =============================================================================

def aggregate_values(values: np.ndarray, method: AggregationType, percentile: float) -> float:
    """Apply aggregation method to an array of values."""
    if method == 'max':
        return np.max(values)
    elif method == 'mean':
        return np.mean(values)
    elif method == 'pq':
        return np.percentile(values, percentile)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_window_scores(model, dataloader, cfg, device, config: ScoringConfig):
    """
    Compute window-level anomaly scores using configurable aggregation.
    
    Args:
        model: ScoreBERT model
        dataloader: DataLoader for test data
        cfg: Configuration
        device: Torch device
        config: ScoringConfig specifying aggregation methods
        
    Returns:
        window_scores: Array of anomaly scores for each window
        score_parts: Array of score_part values for each window
        event_parts: Array of event_part values for each window
    """
    model.eval()
    eventid_dim = cfg.network.encoder.eventid_dim
    
    window_scores = []
    score_parts = []
    event_parts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing scores", leave=False):
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape
            
            # Extract components
            target_eventid = batch[:, :, :eventid_dim]
            target_score = batch[:, :, eventid_dim:eventid_dim+1]
            score_missing = batch[:, :, eventid_dim+1]
            
            # Forward pass (no masking)
            pred_score, pred_eventid = model(batch)
            
            # Compute per-token errors (raw, not normalized)
            raw_score_err = (pred_score.squeeze(-1) - target_score.squeeze(-1)) ** 2
            event_context_err = ((pred_eventid - target_eventid) ** 2).mean(dim=-1)
            
            # Compute window scores for each sample
            for i in range(batch_size):
                valid_mask = (score_missing[i] == 0)
                n_valid = valid_mask.sum().item()
                
                # Event part (all tokens)
                event_vals = event_context_err[i].cpu().numpy()
                event_part = aggregate_values(event_vals, config.event_agg, config.percentile)
                
                # Score part (only valid tokens)
                if n_valid > 0:
                    score_vals = raw_score_err[i][valid_mask].cpu().numpy()
                    score_part = aggregate_values(score_vals, config.score_agg, config.percentile)
                else:
                    score_part = 0.0
                
                # Compute window score
                window_score = score_part + config.gamma * event_part
                
                window_scores.append(window_score)
                score_parts.append(score_part)
                event_parts.append(event_part)
    
    return np.array(window_scores), np.array(score_parts), np.array(event_parts)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_timeline(normal_scores, abnormal_scores, output_dir, config: ScoringConfig):
    """Plot timeline of window scores with normal/abnormal labels."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    n_normal = len(normal_scores)
    n_abnormal = len(abnormal_scores)
    
    x_normal = np.arange(n_normal)
    x_abnormal = np.arange(n_normal, n_normal + n_abnormal)
    
    ax.scatter(x_normal, normal_scores, c='blue', alpha=0.5, s=10, label='Normal')
    ax.scatter(x_abnormal, abnormal_scores, c='red', alpha=0.5, s=10, label='Abnormal')
    
    ax.axvline(n_normal, color='gray', linestyle='--', alpha=0.7, label='Label boundary')
    
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'Window-level Anomaly Scores\n{config.get_formula_string()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timeline_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Timeline plot saved to: {output_dir / 'timeline_v1.png'}")


def plot_score_components(normal_data, abnormal_data, output_dir, config: ScoringConfig):
    """Plot score components (score_part and event_part) separately."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_normal = len(normal_data['score_part'])
    n_abnormal = len(abnormal_data['score_part'])
    
    x_normal = np.arange(n_normal)
    x_abnormal = np.arange(n_normal, n_normal + n_abnormal)
    
    # Score part
    axes[0].scatter(x_normal, normal_data['score_part'], c='blue', alpha=0.5, s=10, label='Normal')
    axes[0].scatter(x_abnormal, abnormal_data['score_part'], c='red', alpha=0.5, s=10, label='Abnormal')
    axes[0].axvline(n_normal, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Window Index')
    axes[0].set_ylabel(f'{config.get_score_label()}(Score Error)')
    axes[0].set_title(f'score_part = {config.get_score_label()}(raw_score_err)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Event part
    axes[1].scatter(x_normal, normal_data['event_part'], c='blue', alpha=0.5, s=10, label='Normal')
    axes[1].scatter(x_abnormal, abnormal_data['event_part'], c='red', alpha=0.5, s=10, label='Abnormal')
    axes[1].axvline(n_normal, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Window Index')
    axes[1].set_ylabel(f'{config.get_event_label()}(Event Error)')
    axes[1].set_title(f'event_part = {config.get_event_label()}(event_context_err)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_components_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score components plot saved to: {output_dir / 'score_components_v1.png'}")


def plot_roc_pr_curves(labels, scores, output_dir):
    """Plot ROC curve and PR curve, return AUC values."""
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(recall, precision, 'r-', linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_pr_curves_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC/PR curves saved to: {output_dir / 'roc_pr_curves_v1.png'}")
    
    return roc_auc, pr_auc


def compute_metrics_at_thresholds(labels, scores, n_thresholds=100):
    """Compute P, R, F1, TPR, FPR at various thresholds."""
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    
    results = []
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
        })
    
    return results


def plot_metrics_curves(metrics, output_dir):
    """Plot P, R, F1, TPR, FPR vs threshold."""
    thresholds = [m['threshold'] for m in metrics]
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    tpr = [m['tpr'] for m in metrics]
    fpr = [m['fpr'] for m in metrics]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(thresholds, precision, 'b-', linewidth=1.5, label='Precision')
    axes[0].plot(thresholds, recall, 'g-', linewidth=1.5, label='Recall')
    axes[0].plot(thresholds, f1, 'r-', linewidth=2, label='F1')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision / Recall / F1 vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    best_idx = np.argmax(f1)
    best_f1 = f1[best_idx]
    best_thresh = thresholds[best_idx]
    axes[0].axvline(best_thresh, color='red', linestyle='--', alpha=0.5)
    axes[0].annotate(f'Best F1={best_f1:.3f}\n@{best_thresh:.2f}', 
                     xy=(best_thresh, best_f1), fontsize=9)
    
    axes[1].plot(thresholds, tpr, 'g-', linewidth=1.5, label='TPR (Recall)')
    axes[1].plot(thresholds, fpr, 'r-', linewidth=1.5, label='FPR')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Rate')
    axes[1].set_title('TPR / FPR vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_curves_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics curves saved to: {output_dir / 'metrics_curves_v1.png'}")
    
    return best_thresh, best_f1


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate(
    model_dir,
    gamma: float = 1.0,
    score_agg: AggregationType = 'max',
    event_agg: AggregationType = 'pq',
    percentile: float = 95
):
    """
    Main evaluation function.
    
    Args:
        model_dir: Path to trained model directory
        gamma: Weight for event error (default: 1.0)
        score_agg: Aggregation method for score error ('max', 'mean', 'pq')
        event_agg: Aggregation method for event error ('max', 'mean', 'pq')
        percentile: Percentile value when using 'pq' aggregation
    """
    # Create scoring configuration
    config = ScoringConfig(
        score_agg=score_agg,
        event_agg=event_agg,
        percentile=percentile,
        gamma=gamma
    )
    
    model_dir = Path(model_dir)
    output_dir = model_dir / 'eval_v1'
    output_dir.mkdir(exist_ok=True)
    
    # Load model and config
    model, cfg, device = load_model_and_config(model_dir)
    
    # Data paths
    data_dir = Path(cfg.dataset.data_dir)
    normal_path = data_dir / 'test_normal.npz'
    abnormal_path = data_dir / 'test_abnormal.npz'
    
    print(f"\nLoading test data...")
    print(f"  Normal: {normal_path}")
    print(f"  Abnormal: {abnormal_path}")
    
    # Load datasets
    normal_dataset = ScoreBERTDataset(normal_path)
    abnormal_dataset = ScoreBERTDataset(abnormal_path)
    
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        shuffle=False,
        num_workers=cfg.default.num_workers,
    )
    abnormal_loader = DataLoader(
        abnormal_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        shuffle=False,
        num_workers=cfg.default.num_workers,
    )
    
    print(f"  Normal windows: {len(normal_dataset)}")
    print(f"  Abnormal windows: {len(abnormal_dataset)}")
    
    # Compute window scores
    print(f"\nComputing anomaly scores...")
    print(f"  Scoring: {config.get_formula_string()}")
    
    normal_scores, normal_score_part, normal_event_part = compute_window_scores(
        model, normal_loader, cfg, device, config
    )
    abnormal_scores, abnormal_score_part, abnormal_event_part = compute_window_scores(
        model, abnormal_loader, cfg, device, config
    )
    
    print(f"  Normal scores: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
    print(f"  Abnormal scores: mean={abnormal_scores.mean():.4f}, std={abnormal_scores.std():.4f}")
    
    # Create labels (0=normal, 1=abnormal)
    labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(abnormal_scores))
    ])
    scores = np.concatenate([normal_scores, abnormal_scores])
    
    # Generate plots
    print(f"\nGenerating plots...")
    plot_timeline(normal_scores, abnormal_scores, output_dir, config)
    
    normal_data = {'score_part': normal_score_part, 'event_part': normal_event_part}
    abnormal_data = {'score_part': abnormal_score_part, 'event_part': abnormal_event_part}
    plot_score_components(normal_data, abnormal_data, output_dir, config)
    
    roc_auc, pr_auc = plot_roc_pr_curves(labels, scores, output_dir)
    
    metrics = compute_metrics_at_thresholds(labels, scores)
    best_thresh, best_f1 = plot_metrics_curves(metrics, output_dir)
    
    # Save results
    results = {
        'scoring_config': {
            'score_agg': config.score_agg,
            'event_agg': config.event_agg,
            'percentile': config.percentile,
            'gamma': config.gamma,
        },
        'formula': config.get_formula_string(),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': best_thresh,
        'best_f1': best_f1,
        'n_normal': len(normal_scores),
        'n_abnormal': len(abnormal_scores),
        'normal_score_mean': float(normal_scores.mean()),
        'normal_score_std': float(normal_scores.std()),
        'abnormal_score_mean': float(abnormal_scores.mean()),
        'abnormal_score_std': float(abnormal_scores.std()),
    }
    
    with open(output_dir / 'eval_results_v1.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Scoring: {config.get_formula_string()}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Best F1: {best_f1:.4f} (threshold={best_thresh:.4f})")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")
    
    return results


# =============================================================================
# Entry Points
# =============================================================================

def run_from_notebook(
    model_dir: str,
    gamma: float = 1.0,
    score_agg: AggregationType = 'max',
    event_agg: AggregationType = 'pq',
    percentile: float = 95,
):
    """
    Jupyter notebook helper.
    
    Args:
        model_dir: Path to trained model directory
        gamma: Weight for event context error (default: 1.0)
        score_agg: Aggregation for score error (default: 'max')
        event_agg: Aggregation for event error (default: 'pq')
        percentile: Percentile value for 'pq' aggregation (default: 95)
    """
    return evaluate(
        model_dir,
        gamma=gamma,
        score_agg=score_agg,
        event_agg=event_agg,
        percentile=percentile
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ScoreBERT Evaluation (V1)')
    parser.add_argument('model_dir', type=str, help='Path to trained model directory')
    parser.add_argument('--gamma', type=float, default=1.0, 
                        help='Weight for event context error (default: 1.0)')
    parser.add_argument('--score-agg', type=str, default='max', choices=['max', 'mean', 'pq'],
                        help='Aggregation method for score error (default: max)')
    parser.add_argument('--event-agg', type=str, default='pq', choices=['max', 'mean', 'pq'],
                        help='Aggregation method for event error (default: pq)')
    parser.add_argument('--percentile', type=float, default=95, 
                        help='Percentile for pq aggregation (default: 95)')
    
    args = parser.parse_args()
    
    evaluate(
        args.model_dir,
        gamma=args.gamma,
        score_agg=args.score_agg,
        event_agg=args.event_agg,
        percentile=args.percentile
    )


if __name__ == '__main__':
    main()
