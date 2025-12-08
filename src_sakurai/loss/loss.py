import torch
import torch.nn as nn
from .maskloss import MaskLoss
from .hypersphereloss import HyperSphereLoss


class LossList(nn.ModuleList):
    def __init__(self, *args: nn.Module):
        super().__init__(args)

    def forward(self, output, data):
        loss_list = []
        N = data[next(iter(data))].shape[0]
        for module in self:
            loss = module(output, data)
            loss_list.append(loss.unsqueeze(0))
        return torch.cat(loss_list), N


class LossCalculate(nn.Module):

    def __init__(self, loss_bias, loss_list):
        super().__init__()
        self.loss = LossList(*loss_list)
        self.loss_bias = loss_bias

    def forward(self, output, data):
        loss, N = self.loss(output, data)
        loss *= self.loss_bias

        return loss, N


def suggest_loss(cfg, device, phase):
    loss_list = []
    loss_name = ["Total"]
    loss_bias = []
    if "mask" in cfg.loss:
        mask = MaskLoss()
        loss_list.append(mask)
        loss_name.append("Mask")
        loss_bias.append(cfg.loss.mask.bias)
    if "hypersphere" in cfg.loss:
        hypersphere = HyperSphereLoss()
        loss_list.append(hypersphere)
        loss_name.append("HyperSphere")
        loss_bias.append(cfg.loss.hypersphere.bias)

    if len(loss_list) == 0:
        raise ValueError("Loss does not exist !!!")
    if phase == "train":
        loss_name_list = [f"Train{n}" for n in loss_name]
        loss_name_list += [f"Val{n}" for n in loss_name]
    else:
        loss_name_list = [f"Normal{n}" for n in loss_name]
        loss_name_list += [f"Abnormal{n}" for n in loss_name]

    loss_bias = torch.tensor(loss_bias, device=device)

    criterion = LossCalculate(loss_bias=loss_bias, loss_list=loss_list)

    return criterion, loss_name_list
