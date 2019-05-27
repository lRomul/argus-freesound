import torch
import random
import numpy as np

from src.transforms import OneOf, UseWithProb, Compose


class PadToSize:
    def __init__(self, size, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = size
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[0] < self.size:
            padding = self.size - signal.shape[0]
            pad_width = ((0, padding), (0, 0))
            if self.mode == 'constant':
                signal = np.pad(signal, pad_width, 'constant')
            else:
                signal = np.pad(signal, pad_width, 'wrap')
        return signal


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):

        if signal.shape[0] > self.size:
            start = (signal.shape[0] - self.size) // 2
            return signal[start: start + self.size]
        else:
            return signal


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        start = random.randint(0, signal.shape[0] - self.size)
        return signal[start: start + self.size]


class ToTensor:
    def __call__(self, probs):
        probs = torch.from_numpy(probs)
        return probs


def get_transforms(train, size):
    if train:
        transforms = Compose([
            PadToSize(size, mode='wrap'),
            RandomCrop(size),
            ToTensor()
        ])
    else:
        transforms = Compose([
            PadToSize(size),
            CenterCrop(size),
            ToTensor()
        ])
    return transforms
