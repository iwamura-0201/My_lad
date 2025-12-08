import torch
import torch.nn as nn


class deeplog(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, vocab_size, embedding_dim=None
    ):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, embedding_dim):
        super(Deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            self.embedding_dim, hidden_size, num_layers, batch_first=True
        )
        self.fc0 = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, device):
        input0 = features[0]
        embed0 = self.embedding(input0)
        h0 = torch.zeros(self.num_layers, embed0.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, embed0.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(embed0, (h0, c0))
        out0 = self.fc0(out[:, -1, :])
        return out0
