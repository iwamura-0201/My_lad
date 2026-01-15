"""
ScoreBERT Training Script

Train a ScoreBERT model for self-supervised learning on log sequences.

Features:
- Config-based settings using OmegaConf (similar to setup_config in util.py)
- NPZ data loading
- Random masking (mask_ratio=0.15)
- Dynamic loss normalization
- Checkpointing and logging

Usage:
    python train_score_bert.py score_bert
    python train_score_bert.py score_bert default.epochs=100 optimizer.hp.batch_size=64
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model.logbert.score_bert import ScoreBERT
from loss.score_bert_loss import ScoreBERTLoss


def setup_score_bert_config(config_file_name: str, override_args: list[str]):
    """
    Load config from YAML file with optional CLI overrides.
    Similar to util.py/setup_config but simplified for ScoreBERT.
    
    Args:
        config_file_name: Name of the config file (without .yaml extension)
        override_args: List of CLI override arguments (e.g., ['default.epochs=100'])
    
    Returns:
        cfg: OmegaConf config object
    """
    config_file_path = f"conf/{config_file_name}.yaml"
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise FileNotFoundError(f"No YAML file found! (path={config_file_path})")

    # Merge CLI overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args_list=override_args))
    
    # Generate output directory path
    if "out_dir" not in cfg:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir_path = (
            f"{cfg.default.dir_name}/"
            + f"{cfg.network.ver}/"
            + f"hidden_{cfg.network.encoder.hidden_size}/"
            + f"layers_{cfg.network.encoder.n_layers}/"
            + f"r_seed_{cfg.default.r_seed}/"
            + f"{timestamp}/"
        )
    else:
        output_dir_path = f"{cfg.out_dir}"
    
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Add computed values to config
    cfg = OmegaConf.merge(cfg, {"out_dir": output_dir_path})
    cfg = OmegaConf.merge(cfg, {"execute_config_name": config_file_name})
    cfg = OmegaConf.merge(cfg, {"override_cmd": override_args})
    
    # Save final config
    with open(os.path.join(output_dir_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    return cfg


class ScoreBERTDataset(Dataset):
    """
    Dataset for ScoreBERT training.
    
    Loads NPZ files containing:
    - data: (n_samples, seq_len, 386) - sequences of tokens
    - labels: (n_samples,) - labels (not used for training)
    - metadata: configuration info
    """

    def __init__(self, npz_path):
        """
        Args:
            npz_path: Path to NPZ file
        """
        data = np.load(npz_path, allow_pickle=True)
        self.data = data['data'].astype(np.float32)
        self.labels = data['labels']
        
        # Parse metadata
        try:
            metadata = data['metadata']
            if len(metadata) > 0:
                self.metadata = json.loads(str(metadata[0]))
            else:
                self.metadata = {}
        except:
            self.metadata = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class Masker:
    """
    Random masking for self-supervised learning.
    
    Masks EventID_emb and score for selected positions.
    - score_missing is NOT masked
    - score is replaced with Gaussian noise (not zeros)
    - EventID_emb is replaced with zeros
    """

    def __init__(self, mask_ratio=0.15, eventid_dim=384, noise_std=1.0):
        """
        Args:
            mask_ratio: Probability of masking each position
            eventid_dim: Dimension of EventID embedding
            noise_std: Standard deviation of Gaussian noise for score masking
        """
        self.mask_ratio = mask_ratio
        self.eventid_dim = eventid_dim
        self.noise_std = noise_std

    def __call__(self, x):
        """
        Apply random masking to input.
        
        Args:
            x: (batch, seq_len, input_dim) input tensor
            
        Returns:
            masked_x: Input with masked positions
            mask_positions: Boolean tensor indicating masked positions
            original_eventid: Original EventID embeddings at masked positions
            original_score: Original scores at masked positions
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Create random mask
        mask_prob = torch.rand(batch_size, seq_len, device=x.device)
        mask_positions = mask_prob < self.mask_ratio

        # Store original values
        original_eventid = x[:, :, :self.eventid_dim].clone()
        original_score = x[:, :, self.eventid_dim:self.eventid_dim+1].clone()
        score_missing = x[:, :, self.eventid_dim+1].clone()

        # Create masked input
        masked_x = x.clone()
        
        # Mask EventID_emb (first 384 dims) with zeros
        mask_expanded_eventid = mask_positions.unsqueeze(-1).expand(-1, -1, self.eventid_dim)
        masked_x[:, :, :self.eventid_dim] = torch.where(
            mask_expanded_eventid,
            torch.zeros_like(masked_x[:, :, :self.eventid_dim]),
            masked_x[:, :, :self.eventid_dim]
        )
        
        # Mask score (dim 384) with Gaussian noise
        noise = torch.randn(batch_size, seq_len, device=x.device) * self.noise_std
        masked_x[:, :, self.eventid_dim] = torch.where(
            mask_positions,
            noise,
            masked_x[:, :, self.eventid_dim]
        )
        
        # DO NOT mask score_missing (dim 385) - keep it intact

        return masked_x, mask_positions, original_eventid, original_score, score_missing


def train_epoch(model, dataloader, optimizer, criterion, masker, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {'score_err': 0, 'eventid_err': 0, 'final_loss': 0}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        batch = batch.to(device)
        
        # Apply masking
        masked_x, mask_positions, original_eventid, original_score, score_missing = masker(batch)
        
        # Forward pass
        pred_score, pred_eventid = model(masked_x)
        
        # Compute loss
        loss, loss_dict = criterion(
            pred_score=pred_score,
            pred_eventid=pred_eventid,
            target_score=original_score,
            target_eventid=original_eventid,
            mask_positions=mask_positions,
            score_missing=score_missing,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'score': f'{loss_dict["score_err"]:.4f}',
            'event': f'{loss_dict["eventid_err"]:.4f}',
        })

    # Average losses
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches

    return avg_loss, loss_components


def validate(model, dataloader, criterion, masker, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    loss_components = {'score_err': 0, 'eventid_err': 0, 'final_loss': 0}
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Apply masking
            masked_x, mask_positions, original_eventid, original_score, score_missing = masker(batch)
            
            # Forward pass
            pred_score, pred_eventid = model(masked_x)
            
            # Compute loss
            loss, loss_dict = criterion(
                pred_score=pred_score,
                pred_eventid=pred_eventid,
                target_score=original_score,
                target_eventid=original_eventid,
                mask_positions=mask_positions,
                score_missing=score_missing,
            )
            
            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
            n_batches += 1

    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches

    return avg_loss, loss_components


# def plot_loss_history(history, output_dir):
#     """
#     Plot loss history showing score_err, eventid_err, and final_loss.
#     Shows both training and validation loss on each graph.
#     
#     Args:
#         history: List of dicts containing loss components per epoch
#         output_dir: Directory to save the plot
#     """
#     epochs = [h['epoch'] for h in history]
#     
#     # Training components
#     train_score_err = [h['train_components']['score_err'] for h in history]
#     train_eventid_err = [h['train_components']['eventid_err'] for h in history]
#     train_final_loss = [h['train_loss'] for h in history]
#     
#     # Validation components
#     val_score_err = [h['val_components']['score_err'] for h in history]
#     val_eventid_err = [h['val_components']['eventid_err'] for h in history]
#     val_final_loss = [h['val_loss'] for h in history]
#     
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     
#     # Plot score_err (train & val)
#     axes[0].plot(epochs, train_score_err, 'b-', linewidth=1.5, label='Train')
#     axes[0].plot(epochs, val_score_err, 'b--', linewidth=1.5, alpha=0.7, label='Val')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Score Error (MSE)')
#     axes[0].set_title('Score Error')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
#     
#     # Plot eventid_err (train & val)
#     axes[1].plot(epochs, train_eventid_err, 'g-', linewidth=1.5, label='Train')
#     axes[1].plot(epochs, val_eventid_err, 'g--', linewidth=1.5, alpha=0.7, label='Val')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('EventID Error (MSE)')
#     axes[1].set_title('EventID Error')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
#     
#     # Plot final_loss (train & val)
#     axes[2].plot(epochs, train_final_loss, 'r-', linewidth=1.5, label='Train')
#     axes[2].plot(epochs, val_final_loss, 'r--', linewidth=1.5, alpha=0.7, label='Val')
#     axes[2].set_xlabel('Epoch')
#     axes[2].set_ylabel('Final Loss (Normalized)')
#     axes[2].set_title('Final Loss')
#     axes[2].legend()
#     axes[2].grid(True, alpha=0.3)
#     
#     plt.tight_layout()
#     
#     # Save as PNG and PDF
#     plt.savefig(output_dir / 'loss_history.png', dpi=150, bbox_inches='tight')
#     plt.savefig(output_dir / 'loss_history.pdf', bbox_inches='tight')
#     plt.close()
#     
#     print(f'Loss history plot saved to: {output_dir / "loss_history.png"}')


def plot_loss_history(history, output_dir):
    """
    Plot loss history with 2x3 grid layout.
    Top row: Training losses (score_err, eventid_err, final_loss)
    Bottom row: Validation losses (score_err, eventid_err, final_loss)
    
    Args:
        history: List of dicts containing loss components per epoch
        output_dir: Directory to save the plot
    """
    epochs = [h['epoch'] for h in history]
    
    # Training components
    train_score_err = [h['train_components']['score_err'] for h in history]
    train_eventid_err = [h['train_components']['eventid_err'] for h in history]
    train_final_loss = [h['train_loss'] for h in history]
    
    # Validation components
    val_score_err = [h['val_components']['score_err'] for h in history]
    val_eventid_err = [h['val_components']['eventid_err'] for h in history]
    val_final_loss = [h['val_loss'] for h in history]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # === Top row: Training losses ===
    # Train score_err
    axes[0, 0].plot(epochs, train_score_err, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Score Error (MSE)')
    axes[0, 0].set_title('Train: Score Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Train eventid_err
    axes[0, 1].plot(epochs, train_eventid_err, 'g-', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('EventID Error (MSE)')
    axes[0, 1].set_title('Train: EventID Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train final_loss
    axes[0, 2].plot(epochs, train_final_loss, 'r-', linewidth=1.5)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Final Loss (Normalized)')
    axes[0, 2].set_title('Train: Final Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    # === Bottom row: Validation losses ===
    # Val score_err
    axes[1, 0].plot(epochs, val_score_err, 'b-', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score Error (MSE)')
    axes[1, 0].set_title('Val: Score Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Val eventid_err
    axes[1, 1].plot(epochs, val_eventid_err, 'g-', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('EventID Error (MSE)')
    axes[1, 1].set_title('Val: EventID Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Val final_loss
    axes[1, 2].plot(epochs, val_final_loss, 'r-', linewidth=1.5)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Final Loss (Normalized)')
    axes[1, 2].set_title('Val: Final Loss')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    plt.savefig(output_dir / 'loss_history.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'loss_history.pdf', bbox_inches='tight')
    plt.close()
    
    print(f'Loss history plot saved to: {output_dir / "loss_history.png"}')


def compute_error_stats(model, full_dataset, cfg, device, output_dir):
    """
    Compute normalization statistics for test phase.
    
    Runs inference on entire training data without masking to compute
    per-token errors and their statistics (μ, σ) for score and eventid.
    
    Args:
        model: Trained ScoreBERT model
        full_dataset: Full training dataset (before split)
        cfg: Configuration object
        device: Torch device
        output_dir: Directory to save statistics
        
    Returns:
        stats: Dict containing μ_score, σ_score, μ_event, σ_event
    """
    print("\nComputing error statistics for normalization...")
    
    model.eval()
    eventid_dim = cfg.network.encoder.eventid_dim
    
    # Create dataloader for full dataset (no shuffling)
    dataloader = DataLoader(
        full_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        shuffle=False,
        num_workers=cfg.default.num_workers,
        pin_memory=True,
    )
    
    all_score_errors = []
    all_eventid_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing stats"):
            batch = batch.to(device)
            batch_size, seq_len, input_dim = batch.shape
            
            # Extract components
            target_eventid = batch[:, :, :eventid_dim]  # (batch, seq, 384)
            target_score = batch[:, :, eventid_dim:eventid_dim+1]  # (batch, seq, 1)
            score_missing = batch[:, :, eventid_dim+1]  # (batch, seq)
            
            # Forward pass (no masking - use original input)
            pred_score, pred_eventid = model(batch)
            
            # Compute per-token errors
            # Score error: MSE per token, exclude score_missing == 1
            score_err = (pred_score.squeeze(-1) - target_score.squeeze(-1)) ** 2  # (batch, seq)
            score_valid_mask = (score_missing == 0)  # (batch, seq)
            valid_score_errors = score_err[score_valid_mask].cpu().numpy()
            all_score_errors.extend(valid_score_errors.tolist())
            
            # EventID error: MSE per token (averaged over 384 dims)
            eventid_err = ((pred_eventid - target_eventid) ** 2).mean(dim=-1)  # (batch, seq)
            all_eventid_errors.extend(eventid_err.reshape(-1).cpu().numpy().tolist())
    
    # Convert to numpy for statistics
    all_score_errors = np.array(all_score_errors)
    all_eventid_errors = np.array(all_eventid_errors)
    
    # Compute statistics
    stats = {
        'mu_score': float(np.mean(all_score_errors)),
        'sigma_score': float(np.std(all_score_errors)),
        'mu_event': float(np.mean(all_eventid_errors)),
        'sigma_event': float(np.std(all_eventid_errors)),
        'n_score_tokens': len(all_score_errors),
        'n_event_tokens': len(all_eventid_errors),
    }
    
    # Save to JSON
    stats_path = output_dir / 'error_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Error statistics saved to: {stats_path}")
    print(f"  μ_score: {stats['mu_score']:.6f}, σ_score: {stats['sigma_score']:.6f}")
    print(f"  μ_event: {stats['mu_event']:.6f}, σ_event: {stats['sigma_event']:.6f}")
    print(f"  Score tokens: {stats['n_score_tokens']:,}, Event tokens: {stats['n_event_tokens']:,}")
    
    return stats


def do_train(cfg):
    """
    Main training function using config.
    
    Args:
        cfg: OmegaConf config object
    """
    # Setup device
    device = cfg.default.device_id
    if not torch.cuda.is_available() and 'cuda' in device:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Output directory: {cfg.out_dir}")
    print(f"Device: {device}")
    print(f"Config: {cfg.execute_config_name}")

    # Set random seed for reproducibility
    if cfg.default.deterministic:
        torch.manual_seed(cfg.default.r_seed)
        np.random.seed(cfg.default.r_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.default.r_seed)

    # Load data
    train_path = Path(cfg.dataset.data_dir) / cfg.dataset.train_file
    print(f"Loading data from {train_path}")
    full_dataset = ScoreBERTDataset(train_path)
    
    # Split into train/val using val_rate
    val_rate = cfg.dataset.sample.get('val_rate', 0.1)
    total_size = len(full_dataset)
    val_size = int(total_size * val_rate)
    train_size = total_size - val_size
    
    # Use generator for reproducibility
    generator = torch.Generator().manual_seed(cfg.default.r_seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        shuffle=True,
        num_workers=cfg.default.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.optimizer.hp.batch_size,
        shuffle=False,
        num_workers=cfg.default.num_workers,
        pin_memory=True,
    )
    
    print(f"Total samples: {total_size}")
    print(f"Training samples: {train_size} ({100*(1-val_rate):.0f}%)")
    print(f"Validation samples: {val_size} ({100*val_rate:.0f}%)")

    # Get input dimension from first sample
    sample = full_dataset[0]
    input_dim = sample.shape[-1]
    seq_len = sample.shape[0]
    print(f"Input dimension: {input_dim}, Sequence length: {seq_len}")

    # Create model from config
    model = ScoreBERT(
        input_dim=cfg.network.encoder.input_dim,
        hidden_size=cfg.network.encoder.hidden_size,
        n_layers=cfg.network.encoder.n_layers,
        n_heads=cfg.network.encoder.n_heads,
        dropout=cfg.network.encoder.dropout,
        max_len=cfg.network.encoder.max_len,
        eventid_dim=cfg.network.encoder.eventid_dim,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create criterion, optimizer, scheduler
    criterion = ScoreBERTLoss(
        epsilon=cfg.loss.epsilon,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.hp.lr,
        weight_decay=cfg.optimizer.hp.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.default.epochs,
        eta_min=1e-6,
    )
    
    # Masker with Gaussian noise for score
    noise_std = cfg.dataset.sample.get('noise_std', 1.0)
    masker = Masker(
        mask_ratio=cfg.dataset.sample.mask_ratio,
        noise_std=noise_std,
    )

    # Training loop
    best_loss = float('inf')
    history = []
    output_dir = Path(cfg.out_dir)

    for epoch in range(1, cfg.default.epochs + 1):
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, criterion, masker, device, epoch
        )
        
        # Validate
        if val_loader:
            val_loss, val_components = validate(
                model, val_loader, criterion, masker, device
            )
        else:
            val_loss, val_components = train_loss, train_components
        
        # Update scheduler
        scheduler.step()
        
        # Log
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}/{cfg.default.epochs} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'LR: {current_lr:.6f}')
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_components': train_components,
            'val_loss': val_loss,
            'val_components': val_components,
            'lr': current_lr,
        })
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': OmegaConf.to_container(cfg),
            }, output_dir / 'best_model.pth')
            print(f'  Saved best model (loss: {best_loss:.4f})')
        
        # Note: Checkpoint and final model saving removed - only best_model is saved

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot loss history
    plot_loss_history(history, output_dir)

    # Compute and save error statistics using best model
    # Load best model weights
    best_checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    compute_error_stats(model, full_dataset, cfg, device, output_dir)

    print(f'\nTraining complete!')
    print(f'Best validation loss: {best_loss:.4f}')
    print(f'Results saved to: {output_dir}')
    
    return best_loss


def main():
    """
    Main entry point.
    
    Usage:
        python train_score_bert.py <config_name> [override_args...]
        
    Examples:
        python train_score_bert.py score_bert
        python train_score_bert.py score_bert default.epochs=100
        python train_score_bert.py score_bert optimizer.hp.batch_size=64 network.encoder.hidden_size=256
    """
    if len(sys.argv) < 2:
        print("Usage: python train_score_bert.py <config_name> [override_args...]")
        print("Example: python train_score_bert.py score_bert default.epochs=100")
        sys.exit(1)
    
    config_name = sys.argv[1]
    override_args = sys.argv[2:]
    
    print(f"Loading config: {config_name}")
    if override_args:
        print(f"Override args: {override_args}")
    
    cfg = setup_score_bert_config(config_name, override_args)
    do_train(cfg)


def run_from_notebook(
    config_name: str,
    override_args: list[str] | None = None,
):
    """
    Jupyter notebook helper to run ScoreBERT training.
    """
    if override_args is None:
        override_args = []

    cfg = setup_score_bert_config(config_name, override_args)
    return do_train(cfg)


if __name__ == '__main__':
    main()
