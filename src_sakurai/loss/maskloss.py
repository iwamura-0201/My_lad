import torch.nn as nn


class MaskLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.nll = nn.NLLLoss(ignore_index=0)

    def forward(self, output, data):
        return self.nll(output["logkey_output"].transpose(1, 2), data["bert_label"])
