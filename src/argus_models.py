import torch
from apex import amp

from argus import Model
from argus.utils import deep_detach

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

    def __init__(self, params):
        super().__init__(params)
        self.nn_module, self.optimizer = amp.initialize(
            self.nn_module, self.optimizer,
            opt_level=params['amp']['opt_level'],
            keep_batchnorm_fp32=params['amp']['keep_batchnorm_fp32'],
            loss_scale=params['amp']['loss_scale']
        )

    def train_step(self, batch)-> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item()
        }
