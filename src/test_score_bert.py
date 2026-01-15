"""
ScoreBERT Evaluation Script

Evaluate a trained ScoreBERT model on test data.

Features:
- Load test_normal and test_abnormal data
- Inference without masking
- Normalized error computation using training statistics
- Window-level anomaly scoring
- ROC-AUC and PR-AUC visualization
- P/R/F1 curves at various thresholds

Usage:
    python test_score_bert.py <model_dir>
    python test_score_bert.py outputs/score_bert/score_bert/hidden_128/layers_3/r_seed_42/20260104_xxx
"""

import json
import sys
from pathlib import Path

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


def load_model_and_config(model_dir):
    """
    Load trained model, config, and error statistics.
    
    Args:
        model_dir: Path to model directory containing best_model.pth, config.yaml, error_stats.json
        
    Returns:
        model: Loaded ScoreBERT model
        cfg: Configuration object
        error_stats: Dict with normalization statistics
        device: Torch device
    """
    model_dir = Path(model_dir)
    
    # Load config
    cfg_path = model_dir / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    
    # Load error statistics
    stats_path = model_dir / 'error_stats.json'
    with open(stats_path, 'r') as f:
        error_stats = json.load(f)
    
    # Setup device
    device = cfg.default.device_id
    if not torch.cuda.is_available() and 'cuda' in device:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Create model
    model = ScoreBERT(
        input_dim=cfg.network.encoder.input_dim,
        hidden_size=cfg.network.encoder.hidden_size,
        n_layers=cfg.network.encoder.n_layers,
        n_heads=cfg.network.encoder.n_heads,
        dropout=cfg.network.encoder.dropout,
        max_len=cfg.network.encoder.max_len,
        eventid_dim=cfg.network.encoder.eventid_dim,
    )
    
    # Load weights
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_dir}")
    print(f"Best epoch: {checkpoint['epoch']}, Best loss: {checkpoint['loss']:.4f}")
    
    return model, cfg, error_stats, device


def compute_window_scores(model, dataloader, cfg, error_stats, device, 
                          beta=1.0, epsilon=1e-8, score_method='mean', topk=10):
    """
    Compute window-level anomaly scores.
    
    Args:
        model: ScoreBERT model
        dataloader: DataLoader for test data
        cfg: Configuration
        error_stats: Normalization statistics
        device: Torch device
        beta: Weight for event error
        epsilon: Small value for numerical stability
        score_method: 'mean' or 'topk' - how to aggregate per-token errors
        topk: Number of top errors to sum (only used when score_method='topk')
        
    Returns:
        window_scores: Array of anomaly scores for each window
    """
    model.eval()
    eventid_dim = cfg.network.encoder.eventid_dim
    
    mu_score = error_stats['mu_score']
    sigma_score = error_stats['sigma_score']
    mu_event = error_stats['mu_event']
    sigma_event = error_stats['sigma_event']
    
    window_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing scores", leave=False):
            batch = batch.to(device)
            batch_size, seq_len, input_dim = batch.shape
            
            # Extract components
            target_eventid = batch[:, :, :eventid_dim]
            target_score = batch[:, :, eventid_dim:eventid_dim+1]
            score_missing = batch[:, :, eventid_dim+1]
            
            # Forward pass (no masking)
            pred_score, pred_eventid = model(batch)
            
            # Compute per-token errors
            score_err = (pred_score.squeeze(-1) - target_score.squeeze(-1)) ** 2
            eventid_err = ((pred_eventid - target_eventid) ** 2).mean(dim=-1)
            
            # Normalize
            score_err_norm = (score_err - mu_score) / (sigma_score + epsilon)
            event_err_norm = (eventid_err - mu_event) / (sigma_event + epsilon)
            
            # Compute window scores
            for i in range(batch_size):
                # Valid tokens for score (score_missing == 0)
                valid_mask = (score_missing[i] == 0)
                n_valid = valid_mask.sum().item()
                
                if score_method == 'mean':
                    # Mean-based scoring
                    event_err_agg = event_err_norm[i].mean().item()
                    
                    if n_valid > 0:
                        score_err_agg = score_err_norm[i][valid_mask].mean().item()
                        window_score = score_err_agg + beta * event_err_agg
                    else:
                        window_score = beta * event_err_agg
                        
                elif score_method == 'topk':
                    # Top-k sum-based scoring
                    # Event error: top-k sum
                    k_event = min(topk, seq_len)
                    event_topk, _ = torch.topk(event_err_norm[i], k_event)
                    event_err_agg = event_topk.sum().item()
                    
                    if n_valid > 0:
                        # Score error: top-k sum (only valid tokens)
                        valid_scores = score_err_norm[i][valid_mask]
                        k_score = min(topk, n_valid)
                        score_topk, _ = torch.topk(valid_scores, k_score)
                        score_err_agg = score_topk.sum().item()
                        window_score = score_err_agg + beta * event_err_agg
                    else:
                        window_score = beta * event_err_agg
                else:
                    raise ValueError(f"Unknown score_method: {score_method}. Use 'mean' or 'topk'.")
                
                window_scores.append(window_score)
    
    return np.array(window_scores)


def plot_timeline(normal_scores, abnormal_scores, output_dir):
    """
    Plot timeline of window scores with normal/abnormal labels.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot normal first, then abnormal
    n_normal = len(normal_scores)
    n_abnormal = len(abnormal_scores)
    
    x_normal = np.arange(n_normal)
    x_abnormal = np.arange(n_normal, n_normal + n_abnormal)
    
    ax.scatter(x_normal, normal_scores, c='blue', alpha=0.5, s=10, label='Normal')
    ax.scatter(x_abnormal, abnormal_scores, c='red', alpha=0.5, s=10, label='Abnormal')
    
    ax.axvline(n_normal, color='gray', linestyle='--', alpha=0.7, label='Label boundary')
    
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Window-level Anomaly Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Timeline plot saved to: {output_dir / 'timeline.png'}")


def plot_roc_pr_curves(labels, scores, output_dir):
    """
    Plot ROC curve and PR curve, return AUC values.
    """
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    
    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR curve
    axes[1].plot(recall, precision, 'r-', linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC/PR curves saved to: {output_dir / 'roc_pr_curves.png'}")
    
    return roc_auc, pr_auc, (fpr, tpr, roc_thresholds), (precision, recall, pr_thresholds)


def compute_metrics_at_thresholds(labels, scores, n_thresholds=100):
    """
    Compute P, R, F1, TPR, FPR at various thresholds.
    """
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
    """
    Plot P, R, F1, TPR, FPR vs threshold.
    """
    thresholds = [m['threshold'] for m in metrics]
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    tpr = [m['tpr'] for m in metrics]
    fpr = [m['fpr'] for m in metrics]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # P, R, F1 vs threshold
    axes[0].plot(thresholds, precision, 'b-', linewidth=1.5, label='Precision')
    axes[0].plot(thresholds, recall, 'g-', linewidth=1.5, label='Recall')
    axes[0].plot(thresholds, f1, 'r-', linewidth=2, label='F1')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision / Recall / F1 vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Best F1
    best_idx = np.argmax(f1)
    best_f1 = f1[best_idx]
    best_thresh = thresholds[best_idx]
    axes[0].axvline(best_thresh, color='red', linestyle='--', alpha=0.5)
    axes[0].annotate(f'Best F1={best_f1:.3f}\n@{best_thresh:.2f}', 
                     xy=(best_thresh, best_f1), fontsize=9)
    
    # TPR, FPR vs threshold
    axes[1].plot(thresholds, tpr, 'g-', linewidth=1.5, label='TPR (Recall)')
    axes[1].plot(thresholds, fpr, 'r-', linewidth=1.5, label='FPR')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Rate')
    axes[1].set_title('TPR / FPR vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics curves saved to: {output_dir / 'metrics_curves.png'}")
    
    return best_thresh, best_f1


def evaluate(model_dir, beta=1.0, score_method='mean', topk=10):
    """
    Main evaluation function.
    
    Args:
        model_dir: Path to trained model directory
        beta: Weight for event error in anomaly score
        score_method: 'mean' or 'topk' - how to aggregate per-token errors
        topk: Number of top errors to sum (only used when score_method='topk')
    """
    model_dir = Path(model_dir)
    output_dir = model_dir / 'eval'
    output_dir.mkdir(exist_ok=True)
    
    # Load model and config
    model, cfg, error_stats, device = load_model_and_config(model_dir)
    
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
    print(f"\nComputing anomaly scores (β={beta}, method={score_method}, topk={topk})...")
    normal_scores = compute_window_scores(
        model, normal_loader, cfg, error_stats, device, 
        beta=beta, score_method=score_method, topk=topk
    )
    abnormal_scores = compute_window_scores(
        model, abnormal_loader, cfg, error_stats, device, 
        beta=beta, score_method=score_method, topk=topk
    )
    
    print(f"  Normal scores: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
    print(f"  Abnormal scores: mean={abnormal_scores.mean():.4f}, std={abnormal_scores.std():.4f}")
    
    # Create labels (0=normal, 1=abnormal)
    labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(abnormal_scores))
    ])
    scores = np.concatenate([normal_scores, abnormal_scores])
    
    # Plot timeline
    print(f"\nGenerating plots...")
    plot_timeline(normal_scores, abnormal_scores, output_dir)
    
    # ROC and PR curves
    roc_auc, pr_auc, roc_data, pr_data = plot_roc_pr_curves(labels, scores, output_dir)
    
    # Metrics at various thresholds
    metrics = compute_metrics_at_thresholds(labels, scores)
    best_thresh, best_f1 = plot_metrics_curves(metrics, output_dir)
    
    # Save results
    results = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': best_thresh,
        'best_f1': best_f1,
        'beta': beta,
        'n_normal': len(normal_scores),
        'n_abnormal': len(abnormal_scores),
        'normal_score_mean': float(normal_scores.mean()),
        'normal_score_std': float(normal_scores.std()),
        'abnormal_score_mean': float(abnormal_scores.mean()),
        'abnormal_score_std': float(abnormal_scores.std()),
    }
    
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Best F1: {best_f1:.4f} (threshold={best_thresh:.4f})")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")
    
    return results


def main():
    """
    Main entry point.
    
    Usage:
        python test_score_bert.py <model_dir> [beta]
        
    Examples:
        python test_score_bert.py outputs/score_bert/score_bert/hidden_128/layers_3/r_seed_42/xxx
        python test_score_bert.py outputs/score_bert/score_bert/hidden_128/layers_3/r_seed_42/xxx 0.5
    """
    if len(sys.argv) < 2:
        print("Usage: python test_score_bert.py <model_dir> [beta]")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    evaluate(model_dir, beta=beta)


def run_from_notebook(
    model_dir: str,
    beta: float = 1.0,
    score_method: str = 'mean',
    topk: int = 10,
):
    """
    Jupyter notebook helper to run ScoreBERT evaluation.
    """
    return evaluate(model_dir, beta=beta, score_method=score_method, topk=topk)


if __name__ == '__main__':
    main()
