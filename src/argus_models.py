from argus import Model
from src.metrics import MultiCategoricalAccuracy, Lwlrap

from src.models.resnet import resnet18


class FreesoundModel(Model):
    nn_module = {
        'resnet18': resnet18
    }
