import torch
from argus import Model
from argus.utils import deep_to, deep_detach

from apex.fp16_utils import FP16_Optimizer

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


class FreesoundApexModel(Model):
    nn_module = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'FeatureExtractor': FeatureExtractor,
        'SimpleKaggle': SimpleKaggle
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params):
        super().__init__(params)
        self.nn_module = self.nn_module.half()
        self.fp16_optimizer = FP16_Optimizer(self.optimizer,
                                             **params['fp16_optimizer'])

    def prepare_batch(self, batch, device):
        input, target = batch
        input = deep_to(input, device,
                        dtype=torch.float16,
                        non_blocking=True)
        target = deep_to(target, device,
                         dtype=torch.float16,
                         non_blocking=True)
        return input, target

    def train_step(self, batch) -> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.fp16_optimizer.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        self.fp16_optimizer.backward(loss)
        self.fp16_optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item()
        }

    def predict(self, input):
        assert self.predict_ready()
        with torch.no_grad():
            if self.nn_module.training:
                self.nn_module.eval()
            input = deep_to(input, self.device,
                            dtype=torch.float16)
            prediction = self.nn_module(input)
            prediction = self.prediction_transform(prediction)
            return prediction
