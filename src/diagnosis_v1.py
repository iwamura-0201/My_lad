"""
ScoreBERT Diagnosis Script (V1)

Diagnostic analysis of ScoreBERT model performance using configurable scoring.

Window-level anomaly score (configurable):
    score_part  = AGG_score(raw_score_err)    # default: max
    event_part  = AGG_event(event_context_err) # default: P95
    window_score = score_part + γ * event_part

Features:
- Configurable aggregation methods for score and event errors
- Window score histograms comparing normal vs abnormal
- Raw error distribution (not normalized)
- Separate AUC evaluation for score_part and event_part
- Missing ratio distribution

Usage:
    python diagnosis_v1.py <model_dir>
    python diagnosis_v1.py <model_dir> --gamma 0.5 --score-agg max --event-agg pq
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.logbert.score_bert import ScoreBERT
from train_score_bert import ScoreBERTDataset


# =============================================================================
# Scoring Configuration
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
    
    Attributes:
        score_agg: Aggregation method for score error ('max', 'mean', 'pq')
        event_agg: Aggregation method for event error ('max', 'mean', 'pq')
        percentile: Percentile value when using 'pq' aggregation
        gamma: Weight for event_part in final score
    """
    score_agg: AggregationType = 'max'
    event_agg: AggregationType = 'pq'
    percentile: float = 95
    gamma: float = 1.0
    
    def get_score_key(self) -> str:
        """Get the stats dictionary key for score aggregation."""
        return f'score_{self.score_agg}'
    
    def get_event_key(self) -> str:
        """Get the stats dictionary key for event aggregation."""
        return f'event_{self.event_agg}'
    
    def get_score_label(self) -> str:
        """Get human-readable label for score aggregation."""
        if self.score_agg == 'pq':
            return f'P{int(self.percentile)}'
        return self.score_agg.upper()
    
    def get_event_label(self) -> str:
        """Get human-readable label for event aggregation."""
        if self.event_agg == 'pq':
            return f'P{int(self.percentile)}'
        return self.event_agg.upper()
    
    def get_formula_string(self) -> str:
        """Get the scoring formula as a string."""
        score_label = self.get_score_label()
        event_label = self.get_event_label()
        return f"window_score = {score_label}(raw_score_err) + {self.gamma} × {event_label}(event_context_err)"


# Default configuration
DEFAULT_CONFIG = ScoringConfig(
    score_agg='max',
    event_agg='pq',
    percentile=95,
    gamma=1.0
)


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_config(model_dir):
    """Load trained model and config (no normalization stats needed for V1)."""
    model_dir = Path(model_dir)
    
    cfg = OmegaConf.load(model_dir / 'config.yaml')
    
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
    
    return model, cfg, device


# =============================================================================
# Score Computation
# =============================================================================

def compute_detailed_scores_raw(model, dataloader, cfg, device, percentile=95):
    """
    Compute detailed per-window statistics using raw (non-normalized) errors.
    
    Computes all aggregation methods (pq, mean, max) to allow flexible scoring.
    
    Returns:
        dict with:
            - score_pq, score_mean, score_max: Score error aggregations
            - event_pq, event_mean, event_max: Event error aggregations
            - missing_ratio: Ratio of missing tokens per window
    """
    model.eval()
    eventid_dim = cfg.network.encoder.eventid_dim
    
    results = {
        # All aggregation methods for flexibility
        'score_pq': [], 'score_mean': [], 'score_max': [],
        'event_pq': [], 'event_mean': [], 'event_max': [],
        # Missing ratio
        'missing_ratio': [],
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
            
            # Compute per-token errors (RAW, not normalized)
            raw_score_err = (pred_score.squeeze(-1) - target_score.squeeze(-1)) ** 2
            event_context_err = ((pred_eventid - target_eventid) ** 2).mean(dim=-1)
            
            for i in range(batch_size):
                valid_mask = (score_missing[i] == 0)
                n_valid = valid_mask.sum().item()
                n_total = seq_len
                
                # Missing ratio
                missing_ratio = 1.0 - (n_valid / n_total)
                results['missing_ratio'].append(missing_ratio)
                
                # Event error aggregations (all tokens, raw values)
                event_vals = event_context_err[i].cpu().numpy()
                results['event_pq'].append(np.percentile(event_vals, percentile))
                results['event_mean'].append(np.mean(event_vals))
                results['event_max'].append(np.max(event_vals))
                
                # Score error aggregations (only valid tokens, raw values)
                if n_valid > 0:
                    score_vals = raw_score_err[i][valid_mask].cpu().numpy()
                    results['score_pq'].append(np.percentile(score_vals, percentile))
                    results['score_mean'].append(np.mean(score_vals))
                    results['score_max'].append(np.max(score_vals))
                else:
                    results['score_pq'].append(0.0)
                    results['score_mean'].append(0.0)
                    results['score_max'].append(0.0)
    
    return {k: np.array(v) for k, v in results.items()}


def compute_window_scores(stats, config: ScoringConfig):
    """
    Compute window scores using the configured scoring formula.
    
    Args:
        stats: Dictionary with aggregated statistics
        config: ScoringConfig specifying aggregation methods and gamma
        
    Returns:
        window_scores: Array of computed window scores
    """
    score_key = config.get_score_key()
    event_key = config.get_event_key()
    
    score_part = stats[score_key]
    event_part = stats[event_key]
    
    return score_part + config.gamma * event_part


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_histograms_raw(normal_stats, abnormal_stats, output_dir, percentile=95):
    """Plot window score histograms for pq, mean, max aggregations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    aggregations = [('pq', f'P{percentile}'), ('mean', 'Mean'), ('max', 'Max')]
    parts = ['score', 'event']
    
    for row, part in enumerate(parts):
        for col, (agg_key, agg_label) in enumerate(aggregations):
            key = f'{part}_{agg_key}'
            ax = axes[row, col]
            
            normal_vals = normal_stats[key]
            abnormal_vals = abnormal_stats[key]
            
            all_vals = np.concatenate([normal_vals, abnormal_vals])
            bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 50)
            
            ax.hist(normal_vals, bins=bins, alpha=0.6, label='Normal', color='blue', density=True)
            ax.hist(abnormal_vals, bins=bins, alpha=0.6, label='Abnormal', color='red', density=True)
            
            ax.set_xlabel(f'{part.capitalize()} Error ({agg_label})')
            ax.set_ylabel('Density')
            ax.set_title(f'{part.capitalize()} Error - {agg_label} (Raw)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'histograms_raw_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Raw histograms saved to: {output_dir / 'histograms_raw_v1.png'}")


def plot_combined_histograms(normal_stats, abnormal_stats, output_dir, config: ScoringConfig):
    """Plot combined window_score histogram using configured formula."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    normal_scores = compute_window_scores(normal_stats, config)
    abnormal_scores = compute_window_scores(abnormal_stats, config)
    
    all_vals = np.concatenate([normal_scores, abnormal_scores])
    bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 50)
    
    ax.hist(normal_scores, bins=bins, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(abnormal_scores, bins=bins, alpha=0.6, label='Abnormal', color='red', density=True)
    
    ax.set_xlabel('Window Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Window Score Distribution\n{config.get_formula_string()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_histograms_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined histogram saved to: {output_dir / 'combined_histograms_v1.png'}")


def plot_missing_ratio(normal_stats, abnormal_stats, output_dir):
    """Plot missing ratio distribution for normal vs abnormal."""
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
    
    ax.axvline(np.mean(normal_missing), color='blue', linestyle='--', alpha=0.8)
    ax.axvline(np.mean(abnormal_missing), color='red', linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_ratio_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Missing ratio plot saved to: {output_dir / 'missing_ratio_v1.png'}")


def collect_all_token_errors_raw(model, dataloader, cfg, device):
    """Collect all per-token RAW errors for histogram plotting."""
    model.eval()
    eventid_dim = cfg.network.encoder.eventid_dim
    
    all_score_err = []
    all_event_err = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting token errors", leave=False):
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape
            
            target_eventid = batch[:, :, :eventid_dim]
            target_score = batch[:, :, eventid_dim:eventid_dim+1]
            score_missing = batch[:, :, eventid_dim+1]
            
            pred_score, pred_eventid = model(batch)
            
            raw_score_err = (pred_score.squeeze(-1) - target_score.squeeze(-1)) ** 2
            event_context_err = ((pred_eventid - target_eventid) ** 2).mean(dim=-1)
            
            for i in range(batch_size):
                valid_mask = (score_missing[i] == 0)
                all_event_err.append(event_context_err[i].cpu().numpy())
                
                if valid_mask.sum().item() > 0:
                    all_score_err.append(raw_score_err[i][valid_mask].cpu().numpy())
    
    return {
        'score_err': np.concatenate(all_score_err) if all_score_err else np.array([]),
        'event_err': np.concatenate(all_event_err) if all_event_err else np.array([]),
    }


def plot_error_raw_histograms(normal_errors, abnormal_errors, output_dir):
    """Plot individual histograms for raw score_err and event_err."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Score Error histogram
    ax = axes[0]
    normal_vals = normal_errors['score_err']
    abnormal_vals = abnormal_errors['score_err']
    
    if len(normal_vals) > 0 and len(abnormal_vals) > 0:
        all_vals = np.concatenate([normal_vals, abnormal_vals])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 100)
        
        ax.hist(normal_vals, bins=bins, alpha=0.6, label=f'Normal (n={len(normal_vals):,})', 
                color='blue', density=True)
        ax.hist(abnormal_vals, bins=bins, alpha=0.6, label=f'Abnormal (n={len(abnormal_vals):,})', 
                color='red', density=True)
        
        ax.axvline(np.mean(normal_vals), color='blue', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(np.mean(abnormal_vals), color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('raw_score_err')
    ax.set_ylabel('Density')
    ax.set_title('Score Error (Raw) Distribution\n' + 
                 f'Normal μ={np.mean(normal_vals):.6f}, Abnormal μ={np.mean(abnormal_vals):.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Event Error histogram
    ax = axes[1]
    normal_vals = normal_errors['event_err']
    abnormal_vals = abnormal_errors['event_err']
    
    if len(normal_vals) > 0 and len(abnormal_vals) > 0:
        all_vals = np.concatenate([normal_vals, abnormal_vals])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 100)
        
        ax.hist(normal_vals, bins=bins, alpha=0.6, label=f'Normal (n={len(normal_vals):,})', 
                color='blue', density=True)
        ax.hist(abnormal_vals, bins=bins, alpha=0.6, label=f'Abnormal (n={len(abnormal_vals):,})', 
                color='red', density=True)
        
        ax.axvline(np.mean(normal_vals), color='blue', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(np.mean(abnormal_vals), color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('event_context_err')
    ax.set_ylabel('Density')
    ax.set_title('Event Error (Raw) Distribution\n' + 
                 f'Normal μ={np.mean(normal_vals):.6f}, Abnormal μ={np.mean(abnormal_vals):.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_raw_histograms_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Raw error histograms saved to: {output_dir / 'error_raw_histograms_v1.png'}")


def plot_score_event_scatter(normal_stats, abnormal_stats, output_dir, config: ScoringConfig):
    """Plot scatter plot of (event_part, score_part) colored by normal/abnormal."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    score_key = config.get_score_key()
    event_key = config.get_event_key()
    
    ax.scatter(
        normal_stats[event_key], normal_stats[score_key],
        c='blue', alpha=0.3, s=15, label=f'Normal (n={len(normal_stats[event_key]):,})'
    )
    ax.scatter(
        abnormal_stats[event_key], abnormal_stats[score_key],
        c='red', alpha=0.3, s=15, label=f'Abnormal (n={len(abnormal_stats[event_key]):,})'
    )
    
    # Centroids
    ax.scatter(np.mean(normal_stats[event_key]), np.mean(normal_stats[score_key]), 
               c='blue', s=200, marker='*', edgecolor='black', linewidth=1.5, 
               label='Normal centroid', zorder=5)
    ax.scatter(np.mean(abnormal_stats[event_key]), np.mean(abnormal_stats[score_key]), 
               c='red', s=200, marker='*', edgecolor='black', linewidth=1.5, 
               label='Abnormal centroid', zorder=5)
    
    ax.set_xlabel(f'{config.get_event_label()}(event_context_err)')
    ax.set_ylabel(f'{config.get_score_label()}(raw_score_err)')
    ax.set_title(f'Window-level Error Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_event_scatter_v1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score-Event scatter plot saved to: {output_dir / 'score_event_scatter_v1.png'}")


# =============================================================================
# AUC Computation
# =============================================================================

def compute_separate_aucs(normal_stats, abnormal_stats, config: ScoringConfig):
    """Compute separate AUC for score_part and event_part using configured aggregations."""
    n_normal = len(normal_stats['score_max'])
    n_abnormal = len(abnormal_stats['score_max'])
    
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    
    score_key = config.get_score_key()
    event_key = config.get_event_key()
    
    # Score part AUC
    score_scores = np.concatenate([normal_stats[score_key], abnormal_stats[score_key]])
    score_auc = roc_auc_score(labels, score_scores)
    
    # Event part AUC
    event_scores = np.concatenate([normal_stats[event_key], abnormal_stats[event_key]])
    event_auc = roc_auc_score(labels, event_scores)
    
    return score_auc, event_auc


def compute_combined_auc(normal_stats, abnormal_stats, config: ScoringConfig):
    """Compute AUC for the combined window score."""
    n_normal = len(normal_stats['score_max'])
    n_abnormal = len(abnormal_stats['score_max'])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    
    normal_scores = compute_window_scores(normal_stats, config)
    abnormal_scores = compute_window_scores(abnormal_stats, config)
    combined_scores = np.concatenate([normal_scores, abnormal_scores])
    
    return roc_auc_score(labels, combined_scores)


# =============================================================================
# Main Diagnosis Function
# =============================================================================

def diagnose(
    model_dir,
    gamma: float = 1.0,
    score_agg: AggregationType = 'max',
    event_agg: AggregationType = 'pq',
    percentile: float = 95
):
    """
    Run full diagnosis on a trained model.
    
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
    output_dir = model_dir / 'diagnosis_v1'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Running diagnosis (V1) for: {model_dir}")
    print(f"  Scoring: {config.get_formula_string()}")
    
    # Load model and config
    model, cfg, device = load_model_and_config(model_dir)
    
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
    print("\nComputing detailed statistics (raw errors)...")
    normal_stats = compute_detailed_scores_raw(model, normal_loader, cfg, device, percentile=percentile)
    abnormal_stats = compute_detailed_scores_raw(model, abnormal_loader, cfg, device, percentile=percentile)
    
    # Collect per-token errors
    print("\nCollecting per-token raw errors...")
    normal_errors = collect_all_token_errors_raw(model, normal_loader, cfg, device)
    abnormal_errors = collect_all_token_errors_raw(model, abnormal_loader, cfg, device)
    
    # Generate plots
    print("\nGenerating diagnosis plots...")
    plot_histograms_raw(normal_stats, abnormal_stats, output_dir, percentile=percentile)
    plot_combined_histograms(normal_stats, abnormal_stats, output_dir, config)
    plot_missing_ratio(normal_stats, abnormal_stats, output_dir)
    plot_error_raw_histograms(normal_errors, abnormal_errors, output_dir)
    plot_score_event_scatter(normal_stats, abnormal_stats, output_dir, config)
    
    # Compute AUCs
    score_auc, event_auc = compute_separate_aucs(normal_stats, abnormal_stats, config)
    combined_auc = compute_combined_auc(normal_stats, abnormal_stats, config)
    
    # Compute window scores
    normal_window_scores = compute_window_scores(normal_stats, config)
    abnormal_window_scores = compute_window_scores(abnormal_stats, config)
    
    # Save results
    results = {
        'scoring_config': {
            'score_agg': config.score_agg,
            'event_agg': config.event_agg,
            'percentile': config.percentile,
            'gamma': config.gamma,
        },
        'formula': config.get_formula_string(),
        'score_part_auc': score_auc,
        'event_part_auc': event_auc,
        'combined_auc': combined_auc,
        'n_normal': len(normal_dataset),
        'n_abnormal': len(abnormal_dataset),
        'normal_window_score_mean': float(np.mean(normal_window_scores)),
        'normal_window_score_std': float(np.std(normal_window_scores)),
        'abnormal_window_score_mean': float(np.mean(abnormal_window_scores)),
        'abnormal_window_score_std': float(np.std(abnormal_window_scores)),
        'normal_missing_ratio_mean': float(np.mean(normal_stats['missing_ratio'])),
        'abnormal_missing_ratio_mean': float(np.mean(abnormal_stats['missing_ratio'])),
    }
    
    with open(output_dir / 'diagnosis_results_v1.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DIAGNOSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nScoring Formula:")
    print(f"  {config.get_formula_string()}")
    print(f"\nAUC Evaluation:")
    print(f"  Score Part ({config.get_score_label()}):  {score_auc:.4f}")
    print(f"  Event Part ({config.get_event_label()}):  {event_auc:.4f}")
    print(f"  Combined (window_score): {combined_auc:.4f}")
    print(f"\nWindow Scores:")
    print(f"  Normal:   mean={np.mean(normal_window_scores):.4f}, std={np.std(normal_window_scores):.4f}")
    print(f"  Abnormal: mean={np.mean(abnormal_window_scores):.4f}, std={np.std(abnormal_window_scores):.4f}")
    print(f"\nMissing Ratio:")
    print(f"  Normal mean:   {np.mean(normal_stats['missing_ratio']):.4f}")
    print(f"  Abnormal mean: {np.mean(abnormal_stats['missing_ratio']):.4f}")
    print(f"{'='*60}")
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
    percentile: float = 95
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
    return diagnose(
        model_dir,
        gamma=gamma,
        score_agg=score_agg,
        event_agg=event_agg,
        percentile=percentile
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ScoreBERT Diagnosis (V1)')
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
    
    diagnose(
        args.model_dir,
        gamma=args.gamma,
        score_agg=args.score_agg,
        event_agg=args.event_agg,
        percentile=args.percentile
    )


if __name__ == '__main__':
    main()
