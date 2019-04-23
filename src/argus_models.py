import torch
from argus import Model

from src.models.resnet import resnet18, resnet34


class FreesoundModel(Model):
    nn_module = {
        'resnet18': resnet18,
        'resnet34': resnet34
    }
    prediction_transform = torch.nn.Sigmoid
