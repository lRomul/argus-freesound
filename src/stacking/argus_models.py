from argus import Model

from src.stacking.models import SimpleLSTM


class StackingModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM
    }
