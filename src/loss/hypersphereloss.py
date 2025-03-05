import torch.nn as nn
import torch


class HyperSphereLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, data):
        return torch.var(output["cls_output"])
