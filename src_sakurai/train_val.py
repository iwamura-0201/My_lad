import torch
from tqdm import tqdm


# Training
def train(cfg, model, device, data_dict, optimizer, criterion):
    n_train = 0
    model.train()
    dataloader = enumerate(data_dict["train"])

    for i, data in tqdm(dataloader):
        data = {key: value.to(device) for key, value in data.items()}
        optimizer.zero_grad()
        with torch.set_grad_enabled(mode=True):
            output = model(data)
            loss, N = criterion(output, data)
            sum_loss = loss.sum()
            sum_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if i == 0:
                train_loss = loss * N
            else:
                train_loss += loss * N
            n_train += N
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
