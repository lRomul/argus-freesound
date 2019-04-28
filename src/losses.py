import torch
from torch.nn import functional as F


def sigmoid_mse_loss(input_logits, target_logits):
    """Takes sigmoid on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_sigmoid = torch.sigmoid(input_logits)
    target_sigmoid = torch.sigmoid(target_logits)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_sigmoid, target_sigmoid, size_average=False) / num_classes
