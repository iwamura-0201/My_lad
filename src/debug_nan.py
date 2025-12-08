#!/usr/bin/env python
"""
NaNの発生箇所を特定するためのデバッグスクリプト
"""
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

from util import (
    setup_config,
    fixed_r_seed,
    setup_device,
    suggest_network,
)
from dataset.dataset_util import suggest_dataloader


def check_nan_in_tensor(tensor, name):
    """テンソル内のNaN/Infをチェック"""
    if tensor is None:
        return False
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        print(f"❌ NaN detected in {name}: {nan_count} values")
        return True
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        print(f"❌ Inf detected in {name}: {inf_count} values")
        return True
    print(f"✅ {name}: OK (min={tensor.min().item():.4f}, max={tensor.max().item():.4f})")
    return False


def debug_forward_pass(model, data, device):
    """モデルの各層をデバッグ"""
    print("\n" + "="*60)
    print("DEBUG: Forward Pass Analysis")
    print("="*60)
    
    bert_input = data["bert_input"]
    time_input = data.get("time_input", None)
    
    print(f"\n[Input Data]")
    check_nan_in_tensor(bert_input, "bert_input")
    if time_input is not None:
        check_nan_in_tensor(time_input, "time_input")
    
    # Mask
    mask = (bert_input > 0).unsqueeze(1).repeat(1, bert_input.size(1), 1).unsqueeze(1)
    print(f"\n[Attention Mask]")
    print(f"  Shape: {mask.shape}")
    print(f"  Non-zero ratio: {mask.float().mean().item():.4f}")
    
    encoder = model.encoder
    
    # 1. Embedding層
    print(f"\n[Embedding Layer]")
    x = encoder.embedding(bert_input, None, time_input)
    check_nan_in_tensor(x, "embedding_output")
    
    # 2. Transformer Blocks
    for i, transformer in enumerate(encoder.transformer_blocks):
        print(f"\n[Transformer Block {i}]")
        
        # Input sublayer (Attention)
        input_sublayer = transformer.input_sublayer
        norm_x = input_sublayer.norm(x)
        check_nan_in_tensor(norm_x, f"  norm_before_attention_{i}")
        
        # Attention
        attention = transformer.attention
        query, key, value = [
            l(norm_x).view(norm_x.size(0), -1, attention.h, attention.d_k).transpose(1, 2)
            for l in attention.linear_layers
        ]
        check_nan_in_tensor(query, f"  query_{i}")
        check_nan_in_tensor(key, f"  key_{i}")
        check_nan_in_tensor(value, f"  value_{i}")
        
        # Attention scores
        import math
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        check_nan_in_tensor(scores, f"  attention_scores_{i}")
        
        # Masked scores
        if mask is not None:
            masked_scores = scores.masked_fill(mask == 0, -1e9)
            check_nan_in_tensor(masked_scores, f"  masked_scores_{i}")
            print(f"    Masked score range: {masked_scores.min().item():.2e} to {masked_scores.max().item():.2e}")
        
        # Softmax
        import torch.nn.functional as F
        p_attn = F.softmax(masked_scores, dim=-1)
        check_nan_in_tensor(p_attn, f"  softmax_attention_{i}")
        
        # Full transformer block
        x = transformer.forward(x, mask)
        check_nan_in_tensor(x, f"  transformer_output_{i}")
        
        if check_nan_in_tensor(x, f"  *** STOPPING: NaN at block {i}"):
            break
    
    # 3. Final output
    print(f"\n[Final Outputs]")
    cls_output = x[:, 0]
    check_nan_in_tensor(cls_output, "cls_output")
    
    # Masked LM head
    logkey_output = model.mask_lm(x)
    check_nan_in_tensor(logkey_output, "logkey_output")
    
    return x


def main():
    config_file_name = "bert/test"
    override_args = ["default.epochs=1"]
    
    cfg = setup_config(config_file_name, override_args)
    device = setup_device(cfg)
    fixed_r_seed(cfg)
    
    print(f"Device: {device}")
    print(f"Vocab size: {cfg.dataset.vocab.vocab_size}")
    print(f"Seq len: {cfg.dataset.sample.seq_len}")
    
    # Load data
    data_dict = suggest_dataloader(cfg)
    
    # Load model
    model = suggest_network(cfg)
    model.to(device)
    model.train()
    
    # Get first batch
    train_loader = data_dict["train"]
    
    for i, data in enumerate(train_loader):
        if i >= 3:  # Check first 3 batches
            break
            
        print(f"\n{'='*60}")
        print(f"Batch {i}")
        print(f"{'='*60}")
        
        data = {key: value.to(device) for key, value in data.items()}
        
        debug_forward_pass(model, data, device)
        
        # Forward pass
        with torch.set_grad_enabled(True):
            output = model(data)
            
            print("\n[Model Output Summary]")
            check_nan_in_tensor(output["logkey_output"], "model.logkey_output")
            check_nan_in_tensor(output["cls_output"], "model.cls_output")


if __name__ == "__main__":
    main()
