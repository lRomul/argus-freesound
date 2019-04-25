import torch
from argus import Model

from src.models.resnet import resnet18, resnet34
from src.models.feature_extractor import FeatureExtractor
from src.models.simple_kaggle import SimpleKaggle


class FreesoundModel(Model):
    nn_module = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'FeatureExtractor': FeatureExtractor,
        'SimpleKaggle': SimpleKaggle
    }
    prediction_transform = torch.nn.Sigmoid
