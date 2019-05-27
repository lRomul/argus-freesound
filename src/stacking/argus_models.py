import torch
from argus import Model

from src.stacking.models import FCNet


class StackingModel(Model):
    nn_module = {
        'FCNet': FCNet
    }
    prediction_transform = torch.nn.Sigmoid
