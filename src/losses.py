import torch
from torch import nn


def lq_loss(y_pred, y_true, q):
    eps = 1e-7
    mul = y_pred * y_true
    loss, _ = torch.max(mul, dim=1)
    loss = (1 - (loss + eps) ** q) / q
    return loss.sum()


class LqLoss(nn.Module):
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, output, target):
        return lq_loss(output, target, self.q)


class OnlyNoisyLqLoss(nn.Module):
    def __init__(self, q=0.5):
        self.lq_loss = LqLoss(q=q)
        self.bce_loss = nn.BCEWithLogitsLoss(reduce='sum')

    def forward(self, output, target):
        target, noisy = target

        noisy_indexes = noisy.nonzero().squeeze(1)
        curated_indexes = (noisy == 0).nonzero().squeeze(1)

        if noisy_indexes.shape[0] > 0:
            noisy_target = target[noisy_indexes]
            noisy_output = output[noisy_indexes]
            noisy_output = torch.sigmoid(noisy_output)
            noisy_loss = self.lq_loss(noisy_output, noisy_target)
        else:
            noisy_loss = 0

        if curated_indexes.shape[0] > 0:
            curated_target = target[curated_indexes]
            curated_output = output[curated_indexes]
            curated_loss = self.bce_loss(curated_output, curated_target)
        else:
            curated_loss = 0

        loss = (noisy_loss + curated_loss) / output.shape[0]
        return loss
