from cnn_finetune import make_model

from argus import Model
from src.metrics import MultiCategoricalAccuracy


class CnnFinetune(Model):
    nn_module = make_model
