import torch
from torch import nn
import torch.nn.functional as F


def lq_loss(y_pred, y_true, q):
    eps = 1e-7
    loss = y_pred * y_true
    # loss, _ = torch.max(loss, dim=1)
    loss = (1 - (loss + eps) ** q) / q
    return loss.mean()


class LqLoss(nn.Module):
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, output, target):
        output = torch.sigmoid(output)
        return lq_loss(output, target, self.q)


def l_soft(y_pred, y_true, beta):
    eps = 1e-7

    y_pred = torch.clamp(y_pred, eps, 1.0)

    # (1) dynamically update the targets based on the current state of the model:
    # bootstrapped target tensor
    # use predicted class proba directly to generate regression targets
    with torch.no_grad():
        y_true_update = beta * y_true + (1 - beta) * y_pred

    # (2) compute loss as always
    loss = F.binary_cross_entropy(y_pred, y_true_update)
    return loss


class LSoftLoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        output = torch.sigmoid(output)
        return l_soft(output, target, self.beta)


class NoisyCuratedLoss(nn.Module):
    def __init__(self, noisy_loss, curated_loss,
                 noisy_weight=0.5, curated_weight=0.5):
        super().__init__()
        self.noisy_loss = noisy_loss
        self.curated_loss = curated_loss
        self.noisy_weight = noisy_weight
        self.curated_weight = curated_weight

    def forward(self, output, target, noisy):
        batch_size = target.shape[0]

        noisy_indexes = noisy.nonzero().squeeze(1)
        curated_indexes = (noisy == 0).nonzero().squeeze(1)

        noisy_len = noisy_indexes.shape[0]
        if noisy_len > 0:
            noisy_target = target[noisy_indexes]
            noisy_output = output[noisy_indexes]
            noisy_loss = self.noisy_loss(noisy_output, noisy_target)
            noisy_loss = noisy_loss * (noisy_len / batch_size)
        else:
            noisy_loss = 0

        curated_len = curated_indexes.shape[0]
        if curated_len > 0:
            curated_target = target[curated_indexes]
            curated_output = output[curated_indexes]
            curated_loss = self.curated_loss(curated_output, curated_target)
            curated_loss = curated_loss * (curated_len / batch_size)
        else:
            curated_loss = 0

        loss = noisy_loss * self.noisy_weight
        loss += curated_loss * self.curated_weight
        return loss


class OnlyNoisyLqLoss(nn.Module):
    def __init__(self, q=0.5,
                 noisy_weight=0.5,
                 curated_weight=0.5):
        super().__init__()
        lq = LqLoss(q=q)
        bce = nn.BCEWithLogitsLoss()
        self.loss = NoisyCuratedLoss(lq, bce,
                                     noisy_weight,
                                     curated_weight)

    def forward(self, output, target, noisy):
        return self.loss(output, target, noisy)


class OnlyNoisyLSoftLoss(nn.Module):
    def __init__(self, beta,
                 noisy_weight=0.5,
                 curated_weight=0.5):
        super().__init__()
        soft = LSoftLoss(beta)
        bce = nn.BCEWithLogitsLoss()
        self.loss = NoisyCuratedLoss(soft, bce,
                                     noisy_weight,
                                     curated_weight)

    def forward(self, output, target, noisy):
        return self.loss(output, target, noisy)


class BCEMaxOutlierLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target, noisy):
        loss = F.binary_cross_entropy_with_logits(output, target,
                                                  reduction='none')
        loss = loss.mean(dim=1)

        with torch.no_grad():
            outlier_mask = loss > self.alpha * loss.max()
            outlier_mask = outlier_mask * noisy
            outlier_idx = (outlier_mask == 0).nonzero().squeeze(1)

        loss = loss[outlier_idx].mean()
        return loss
