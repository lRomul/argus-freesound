import torch

from argus.metrics.metric import Metric


class MultiCategoricalAccuracy(Metric):
    name = 'multi_accuracy'
    better = 'max'

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        pred = step_output['prediction']
        trg = step_output['target']
        pred = (pred > self.threshold).to(torch.float32)
        correct = torch.eq(pred, trg).all(dim=1).view(-1)
        self.correct += torch.sum(correct).item()
        self.count += correct.shape[0]

    def compute(self):
        if self.count == 0:
            raise Exception('Must be at least one example for computation')
        return self.correct / self.count
