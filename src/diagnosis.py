"""
ScoreBERT Diagnosis Script

Diagnostic analysis of ScoreBERT model performance.

Features:
- Window score histograms comparing normal vs abnormal (mean/p95/max aggregation)
- Missing ratio distribution for normal vs abnormal
- Separate AUC evaluation for score_part and event_part

Usage:
    python diagnosis.py <model_dir>
    python diagnosis.py outputs/score_bert/score_bert/hidden_128/layers_3/r_seed_42/xxx
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.logbert.score_bert import ScoreBERT
from train_score_bert import ScoreBERTDataset


def load_model_and_config(model_dir):
    """Load trained model, config, and error statistics."""
    model_dir = Path(model_dir)
    
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    with open(model_dir / 'error_stats.json', 'r') as f:
        error_stats = json.load(f)
    
    device = cfg.default.device_id
    if not torch.cuda.is_available() and 'cuda' in device:
        device = 'cpu'
    
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
    
    return model, cfg, error_stats, device


def compute_detailed_scores(model, dataloader, cfg, error_stats, device, epsilon=1e-8):
    """
    Compute detailed per-window statistics.
    
    Returns:
        dict with:
            - score_mean, score_p95, score_max: score_err aggregations
            - event_mean, event_p95, event_max: event_err aggregations
            - missing_ratio: ratio of missing tokens per window
    """
    model.eval()
    eventid_dim = cfg.network.encoder.eventid_dim
    
    mu_score = error_stats['mu_score']
    sigma_score = error_stats['sigma_score']
    mu_event = error_stats['mu_event']
    sigma_event = error_stats['sigma_event']
    
    results = {
        'score_mean': [], 'score_p95': [], 'score_max': [],
        'event_mean': [], 'event_p95': [], 'event_max': [],
        'missing_ratio': [],
        'score_part': [],  # For separate AUC
        'event_part': [],  # For separate AUC
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing", leave=False):
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape
            
            # Extract components
            target_eventid = batch[:, :, :eventid_dim]
            target_score = batch[:, :, eventid_dim:eventid_dim+1]
            score_missing = batch[:, :, eventid_dim+1]
            
            # Forward pass
            pred_score, pred_eventid = model(batch)
            
            # Compute per-token errors
            score_err = (pred_score.squeeze(-1) - target_score.squeeze(-1)) ** 2
            eventid_err = ((pred_eventid - target_eventid) ** 2).mean(dim=-1)
            
            # Normalize
            score_err_norm = (score_err - mu_score) / (sigma_score + epsilon)
            event_err_norm = (eventid_err - mu_event) / (sigma_event + epsilon)
            
            for i in range(batch_size):
                valid_mask = (score_missing[i] == 0)
                n_valid = valid_mask.sum().item()
                n_total = seq_len
                
                # Missing ratio
                missing_ratio = 1.0 - (n_valid / n_total)
                results['missing_ratio'].append(missing_ratio)
                
                # Event error aggregations (all tokens)
                event_vals = event_err_norm[i].cpu().numpy()
                results['event_mean'].append(np.mean(event_vals))
                results['event_p95'].append(np.percentile(event_vals, 95))
                results['event_max'].append(np.max(event_vals))
                results['event_part'].append(np.mean(event_vals))
                
                # Score error aggregations (only valid tokens)
                if n_valid > 0:
                    score_vals = score_err_norm[i][valid_mask].cpu().numpy()
                    results['score_mean'].append(np.mean(score_vals))
                    results['score_p95'].append(np.percentile(score_vals, 95))
                    results['score_max'].append(np.max(score_vals))
                    results['score_part'].append(np.mean(score_vals))
                else:
                    results['score_mean'].append(0.0)
                    results['score_p95'].append(0.0)
                    results['score_max'].append(0.0)
                    results['score_part'].append(0.0)
    
    return {k: np.array(v) for k, v in results.items()}


def plot_histograms(normal_stats, abnormal_stats, output_dir):
    """
    Plot window score histograms for mean/p95/max aggregations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    aggregations = ['mean', 'p95', 'max']
    parts = ['score', 'event']
    
    for row, part in enumerate(parts):
        for col, agg in enumerate(aggregations):
            key = f'{part}_{agg}'
            ax = axes[row, col]
            
            normal_vals = normal_stats[key]
            abnormal_vals = abnormal_stats[key]
            
            # Determine common bin range
            all_vals = np.concatenate([normal_vals, abnormal_vals])
            bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 50)
            
            ax.hist(normal_vals, bins=bins, alpha=0.6, label='Normal', color='blue', density=True)
            ax.hist(abnormal_vals, bins=bins, alpha=0.6, label='Abnormal', color='red', density=True)
            
            ax.set_xlabel(f'{part.capitalize()} Error ({agg})')
            ax.set_ylabel('Density')
            ax.set_title(f'{part.capitalize()} Error - {agg.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histograms saved to: {output_dir / 'histograms.png'}")


def plot_combined_histograms(normal_stats, abnormal_stats, output_dir, beta=1.0):
    """
    Plot combined window_score histograms (score + β * event) for mean/p95/max.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    aggregations = ['mean', 'p95', 'max']
    
    for col, agg in enumerate(aggregations):
        ax = axes[col]
        
        normal_combined = normal_stats[f'score_{agg}'] + beta * normal_stats[f'event_{agg}']
        abnormal_combined = abnormal_stats[f'score_{agg}'] + beta * abnormal_stats[f'event_{agg}']
        
        all_vals = np.concatenate([normal_combined, abnormal_combined])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 50)
        
        ax.hist(normal_combined, bins=bins, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(abnormal_combined, bins=bins, alpha=0.6, label='Abnormal', color='red', density=True)
        
        ax.set_xlabel(f'Window Score ({agg})')
        ax.set_ylabel('Density')
        ax.set_title(f'Window Score (score + {beta}×event) - {agg.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined histograms saved to: {output_dir / 'combined_histograms.png'}")


def plot_missing_ratio(normal_stats, abnormal_stats, output_dir):
    """
    Plot missing ratio distribution for normal vs abnormal.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    normal_missing = normal_stats['missing_ratio']
    abnormal_missing = abnormal_stats['missing_ratio']
    
    bins = np.linspace(0, 1, 30)
    
    ax.hist(normal_missing, bins=bins, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(abnormal_missing, bins=bins, alpha=0.6, label='Abnormal', color='red', density=True)
    
    ax.set_xlabel('Missing Ratio (score_missing=1)')
    ax.set_ylabel('Density')
    ax.set_title('Missing Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.axvline(np.mean(normal_missing), color='blue', linestyle='--', alpha=0.8)
    ax.axvline(np.mean(abnormal_missing), color='red', linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Missing ratio plot saved to: {output_dir / 'missing_ratio.png'}")


def collect_all_token_errors(model, dataloader, cfg, error_stats, device, epsilon=1e-8):
    """
    Collect all per-token normalized errors for histogram plotting.
    
    Returns:
        dict with:
            - score_err_norm: All normalized score errors (only valid tokens)
            - event_err_norm: All normalized event errors (all tokens)
    """
    model.eval()
    eventid_dim = cfg.network.encoder.eventid_dim
    
    mu_score = error_stats['mu_score']
    sigma_score = error_stats['sigma_score']
    mu_event = error_stats['mu_event']
    sigma_event = error_stats['sigma_event']
    
    all_score_err = []
    all_event_err = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting token errors", leave=False):
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape
            
            # Extract components
            target_eventid = batch[:, :, :eventid_dim]
            target_score = batch[:, :, eventid_dim:eventid_dim+1]
            score_missing = batch[:, :, eventid_dim+1]
            
            # Forward pass
            pred_score, pred_eventid = model(batch)
            
            # Compute per-token errors
            score_err = (pred_score.squeeze(-1) - target_score.squeeze(-1)) ** 2
            eventid_err = ((pred_eventid - target_eventid) ** 2).mean(dim=-1)
            
            # Normalize
            score_err_norm = (score_err - mu_score) / (sigma_score + epsilon)
            event_err_norm = (eventid_err - mu_event) / (sigma_event + epsilon)
            
            for i in range(batch_size):
                valid_mask = (score_missing[i] == 0)
                
                # Collect event errors (all tokens)
                all_event_err.append(event_err_norm[i].cpu().numpy())
                
                # Collect score errors (only valid tokens)
                if valid_mask.sum().item() > 0:
                    all_score_err.append(score_err_norm[i][valid_mask].cpu().numpy())
    
    return {
        'score_err_norm': np.concatenate(all_score_err) if all_score_err else np.array([]),
        'event_err_norm': np.concatenate(all_event_err) if all_event_err else np.array([]),
    }


def plot_error_norm_histograms(normal_errors, abnormal_errors, output_dir):
    """
    Plot individual histograms for score_err_norm and event_err_norm (normal vs abnormal).
    
    Args:
        normal_errors: dict with 'score_err_norm' and 'event_err_norm' arrays for normal data
        abnormal_errors: dict with 'score_err_norm' and 'event_err_norm' arrays for abnormal data
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Score Error Norm histogram
    ax = axes[0]
    normal_vals = normal_errors['score_err_norm']
    abnormal_vals = abnormal_errors['score_err_norm']
    
    if len(normal_vals) > 0 and len(abnormal_vals) > 0:
        all_vals = np.concatenate([normal_vals, abnormal_vals])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 100)
        
        ax.hist(normal_vals, bins=bins, alpha=0.6, label=f'Normal (n={len(normal_vals):,})', 
                color='blue', density=True)
        ax.hist(abnormal_vals, bins=bins, alpha=0.6, label=f'Abnormal (n={len(abnormal_vals):,})', 
                color='red', density=True)
        
        # Add mean lines
        ax.axvline(np.mean(normal_vals), color='blue', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(np.mean(abnormal_vals), color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('score_err_norm')
    ax.set_ylabel('Density')
    ax.set_title('Score Error (Normalized) Distribution\n' + 
                 f'Normal μ={np.mean(normal_vals):.3f}, Abnormal μ={np.mean(abnormal_vals):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Event Error Norm histogram
    ax = axes[1]
    normal_vals = normal_errors['event_err_norm']
    abnormal_vals = abnormal_errors['event_err_norm']
    
    if len(normal_vals) > 0 and len(abnormal_vals) > 0:
        all_vals = np.concatenate([normal_vals, abnormal_vals])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 100)
        
        ax.hist(normal_vals, bins=bins, alpha=0.6, label=f'Normal (n={len(normal_vals):,})', 
                color='blue', density=True)
        ax.hist(abnormal_vals, bins=bins, alpha=0.6, label=f'Abnormal (n={len(abnormal_vals):,})', 
                color='red', density=True)
        
        # Add mean lines
        ax.axvline(np.mean(normal_vals), color='blue', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(np.mean(abnormal_vals), color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('event_err_norm')
    ax.set_ylabel('Density')
    ax.set_title('Event Error (Normalized) Distribution\n' + 
                 f'Normal μ={np.mean(normal_vals):.3f}, Abnormal μ={np.mean(abnormal_vals):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_norm_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error norm histograms saved to: {output_dir / 'error_norm_histograms.png'}")


def plot_score_event_scatter(normal_stats, abnormal_stats, output_dir):
    """
    Plot scatter plot of (event_part, score_part) colored by normal/abnormal.
    
    Args:
        normal_stats: dict with 'score_part' and 'event_part' arrays for normal data
        abnormal_stats: dict with 'score_part' and 'event_part' arrays for abnormal data
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normal points (blue)
    ax.scatter(
        normal_stats['event_part'],
        normal_stats['score_part'],
        c='blue', alpha=0.3, s=15, label=f'Normal (n={len(normal_stats["event_part"]):,})'
    )
    
    # Abnormal points (red)
    ax.scatter(
        abnormal_stats['event_part'],
        abnormal_stats['score_part'],
        c='red', alpha=0.3, s=15, label=f'Abnormal (n={len(abnormal_stats["event_part"]):,})'
    )
    
    # Add mean markers
    normal_event_mean = np.mean(normal_stats['event_part'])
    normal_score_mean = np.mean(normal_stats['score_part'])
    abnormal_event_mean = np.mean(abnormal_stats['event_part'])
    abnormal_score_mean = np.mean(abnormal_stats['score_part'])
    
    ax.scatter(normal_event_mean, normal_score_mean, c='blue', s=200, marker='*', 
               edgecolor='black', linewidth=1.5, label='Normal mean', zorder=5)
    ax.scatter(abnormal_event_mean, abnormal_score_mean, c='red', s=200, marker='*', 
               edgecolor='black', linewidth=1.5, label='Abnormal mean', zorder=5)
    
    ax.set_xlabel('event_part (event_err_norm mean)')
    ax.set_ylabel('score_part (score_err_norm mean)')
    ax.set_title('Window-level Error Distribution\n(event_part vs score_part)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_event_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score-Event scatter plot saved to: {output_dir / 'score_event_scatter.png'}")


def compute_separate_aucs(normal_stats, abnormal_stats):
    """
    Compute separate AUC for score_part and event_part.
    """
    n_normal = len(normal_stats['score_part'])
    n_abnormal = len(abnormal_stats['score_part'])
    
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    
    # Score part AUC
    score_scores = np.concatenate([
        normal_stats['score_part'], 
        abnormal_stats['score_part']
    ])
    score_auc = roc_auc_score(labels, score_scores)
    
    # Event part AUC
    event_scores = np.concatenate([
        normal_stats['event_part'], 
        abnormal_stats['event_part']
    ])
    event_auc = roc_auc_score(labels, event_scores)
    
    return score_auc, event_auc


def compute_aggregation_aucs(normal_stats, abnormal_stats, beta=1.0):
    """
    Compute AUC for each aggregation method (mean, p95, max).
    """
    n_normal = len(normal_stats['score_mean'])
    n_abnormal = len(abnormal_stats['score_mean'])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    
    aucs = {}
    for agg in ['mean', 'p95', 'max']:
        combined = np.concatenate([
            normal_stats[f'score_{agg}'] + beta * normal_stats[f'event_{agg}'],
            abnormal_stats[f'score_{agg}'] + beta * abnormal_stats[f'event_{agg}']
        ])
        aucs[agg] = roc_auc_score(labels, combined)
    
    return aucs


def diagnose(model_dir, beta=1.0):
    """
    Run full diagnosis on a trained model.
    
    Args:
        model_dir: Path to trained model directory
        beta: Weight for event error
    """
    model_dir = Path(model_dir)
    output_dir = model_dir / 'diagnosis'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Running diagnosis for: {model_dir}")
    
    # Load model and config
    model, cfg, error_stats, device = load_model_and_config(model_dir)
    
    # Data paths
    data_dir = Path(cfg.dataset.data_dir)
    normal_path = data_dir / 'test_normal.npz'
    abnormal_path = data_dir / 'test_abnormal.npz'
    
    print(f"\nLoading test data...")
    normal_dataset = ScoreBERTDataset(normal_path)
    abnormal_dataset = ScoreBERTDataset(abnormal_path)
    
    normal_loader = DataLoader(
        normal_dataset, batch_size=cfg.optimizer.hp.batch_size,
        shuffle=False, num_workers=cfg.default.num_workers
    )
    abnormal_loader = DataLoader(
        abnormal_dataset, batch_size=cfg.optimizer.hp.batch_size,
        shuffle=False, num_workers=cfg.default.num_workers
    )
    
    print(f"  Normal windows: {len(normal_dataset)}")
    print(f"  Abnormal windows: {len(abnormal_dataset)}")
    
    # Compute detailed statistics
    print("\nComputing detailed statistics...")
    normal_stats = compute_detailed_scores(model, normal_loader, cfg, error_stats, device)
    abnormal_stats = compute_detailed_scores(model, abnormal_loader, cfg, error_stats, device)
    
    # Collect per-token errors for histogram
    print("\nCollecting per-token errors...")
    normal_errors = collect_all_token_errors(model, normal_loader, cfg, error_stats, device)
    abnormal_errors = collect_all_token_errors(model, abnormal_loader, cfg, error_stats, device)
    
    # Generate plots
    print("\nGenerating diagnosis plots...")
    plot_histograms(normal_stats, abnormal_stats, output_dir)
    plot_combined_histograms(normal_stats, abnormal_stats, output_dir, beta=beta)
    plot_missing_ratio(normal_stats, abnormal_stats, output_dir)
    plot_error_norm_histograms(normal_errors, abnormal_errors, output_dir)
    plot_score_event_scatter(normal_stats, abnormal_stats, output_dir)
    
    # Compute separate AUCs
    score_auc, event_auc = compute_separate_aucs(normal_stats, abnormal_stats)
    agg_aucs = compute_aggregation_aucs(normal_stats, abnormal_stats, beta=beta)
    
    # Save results
    results = {
        'beta': beta,
        'score_part_auc': score_auc,
        'event_part_auc': event_auc,
        'aggregation_aucs': agg_aucs,
        'n_normal': len(normal_dataset),
        'n_abnormal': len(abnormal_dataset),
        'normal_missing_ratio_mean': float(np.mean(normal_stats['missing_ratio'])),
        'abnormal_missing_ratio_mean': float(np.mean(abnormal_stats['missing_ratio'])),
    }
    
    with open(output_dir / 'diagnosis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DIAGNOSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nSeparate AUC Evaluation:")
    print(f"  Score Part AUC:  {score_auc:.4f}")
    print(f"  Event Part AUC:  {event_auc:.4f}")
    print(f"\nAggregation Method Comparison (β={beta}):")
    for agg, auc_val in agg_aucs.items():
        print(f"  {agg.upper():5s} AUC: {auc_val:.4f}")
    print(f"\nMissing Ratio:")
    print(f"  Normal mean:   {np.mean(normal_stats['missing_ratio']):.4f}")
    print(f"  Abnormal mean: {np.mean(abnormal_stats['missing_ratio']):.4f}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    
    return results


def run_from_notebook(model_dir: str, beta: float = 1.0):
    """Jupyter notebook helper."""
    return diagnose(model_dir, beta=beta)


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnosis.py <model_dir> [beta]")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    diagnose(model_dir, beta=beta)


if __name__ == '__main__':
    main()
