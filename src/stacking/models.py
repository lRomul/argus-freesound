import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super().__init__()

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class SimpleLSTM(nn.Module):
    def __init__(self, seq_len, input_size, num_classes, p_dropout=0.2, base_size=64):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, base_size*2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(base_size*4, base_size, bidirectional=True, batch_first=True)

        self.attention = Attention(base_size*2, seq_len)

        self.fc1 = nn.Linear(base_size*2, base_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(base_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.attention(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
