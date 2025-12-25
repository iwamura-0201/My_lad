import torch
from tqdm import tqdm


# Training
def train(cfg, model, device, data_dict, optimizer, criterion):
    n_train = 0
    model.train()
    dataloader = enumerate(data_dict["train"])
    nan_batch_count = 0

    for i, data in tqdm(dataloader):
        data = {key: value.to(device) for key, value in data.items()}
        optimizer.zero_grad()
        with torch.set_grad_enabled(mode=True):
            output = model(data)
            
            # NaNガード: 出力にNaNがある場合はスキップ
            if torch.isnan(output["cls_output"]).any() or torch.isnan(output["logkey_output"]).any():
                nan_batch_count += 1
                if nan_batch_count <= 5:
                    print(f"⚠️ NaN detected in batch {i}, skipping...")
                elif nan_batch_count == 6:
                    print("⚠️ Suppressing further NaN warnings...")
                continue
            
            loss, N = criterion(output, data)
            sum_loss = loss.sum()
            
            # NaNガード: lossがNaNの場合はスキップ
            if torch.isnan(sum_loss):
                nan_batch_count += 1
                if nan_batch_count <= 5:
                    print(f"⚠️ NaN loss at batch {i}, skipping...")
                continue
            
            sum_loss.backward()
            
            # 勾配クリッピングを強化（10.0 → 1.0）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 勾配NaNチェック
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                nan_batch_count += 1
                if nan_batch_count <= 5:
                    print(f"⚠️ NaN gradient at batch {i}, skipping...")
                continue
            
            optimizer.step()
            if i == 0:
                train_loss = loss * N
            else:
                train_loss += loss * N
            n_train += N
    
    # Nanガード
    if nan_batch_count > 0:
        print(f"⚠️ Total batches with NaN: {nan_batch_count}")
    
    return train_loss / n_train


# Validation
def val(cfg, model, device, data_dict, criterion):
    model.eval()
    n_val = 0
    dataloader = enumerate(data_dict["val"])
    with torch.no_grad():
        for i, data in tqdm(dataloader):
            data = {key: value.to(device) for key, value in data.items()}
            output = model(data)
            loss, N = criterion(output, data)
            if i == 0:
                val_loss = loss * N
            else:
                val_loss += loss * N
            n_val += N

    return val_loss / n_val
