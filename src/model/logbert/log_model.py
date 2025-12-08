import torch.nn as nn


class LogBertModel(nn.Module):

    def __init__(self, encoder, cfg):
        super().__init__()
        self.result = {
            "logkey_output": None,
            "time_output": None,
            "cls_output": None,
            "cls_fnn_output": None,
        }
        self.encoder = encoder
        self.mask_lm = MaskedLogModel(
            cfg.network.encoder.hidden_size, cfg.dataset.vocab.vocab_size
        )

    def forward(self, data):
        x = self.encoder(data["bert_input"], time_info=data["time_input"])
        self.result["logkey_output"] = self.mask_lm(x)
        self.result["cls_output"] = x[:, 0]
        return self.result


class MaskedLogModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
