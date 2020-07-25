import torch

from argus import Model
from argus.utils import deep_detach, deep_to

from src.models import resnet
from src.models import senet
from src.models.feature_extractor import FeatureExtractor
from src.models.simple_kaggle import SimpleKaggle
from src.models.simple_attention import SimpleAttention
from src.models.skip_attention import SkipAttention
from src.models.aux_skip_attention import AuxSkipAttention
from src.models.rnn_aux_skip_attention import RnnAuxSkipAttention
from src.losses import OnlyNoisyLqLoss, OnlyNoisyLSoftLoss, BCEMaxOutlierLoss
from src import config


class FreesoundModel(Model):
    nn_module = {
        'resnet18': resnet.resnet18,
        'resnet34': resnet.resnet34,
        'FeatureExtractor': FeatureExtractor,
        'SimpleKaggle': SimpleKaggle,
        'se_resnext50_32x4d': senet.se_resnext50_32x4d,
        'SimpleAttention': SimpleAttention,
        'SkipAttention': SkipAttention,
        'AuxSkipAttention': AuxSkipAttention,
        'RnnAuxSkipAttention': RnnAuxSkipAttention
    }
    loss = {
        'OnlyNoisyLqLoss': OnlyNoisyLqLoss,
        'OnlyNoisyLSoftLoss': OnlyNoisyLSoftLoss,
        'BCEMaxOutlierLoss': BCEMaxOutlierLoss
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params):
        super().__init__(params)

        if 'aux' in params:
            self.aux_weights = params['aux']['weights']
        else:
            self.aux_weights = None

        self.use_amp = not config.kernel and 'amp' in params
        if self.use_amp:
            from apex import amp
            self.amp = amp
            self.nn_module, self.optimizer = self.amp.initialize(
                self.nn_module, self.optimizer,
                opt_level=params['amp']['opt_level'],
                keep_batchnorm_fp32=params['amp']['keep_batchnorm_fp32'],
                loss_scale=params['amp']['loss_scale']
            )

    def train_step(self, batch, state) -> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()
        input, target, noisy = deep_to(batch, self.device, non_blocking=True)
        prediction = self.nn_module(input)
        if self.aux_weights is not None:
            loss = 0
            for pred, weight in zip(prediction, self.aux_weights):
                loss += self.loss(pred, target, noisy) * weight
        else:
            loss = self.loss(prediction, target, noisy)
        if self.use_amp:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction[0]),
            'target': target,
            'loss': loss.item(),
            'noisy': noisy
        }

    def val_step(self, batch, state) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()
        with torch.no_grad():
            input, target, noisy = deep_to(batch, self.device, non_blocking=True)
            prediction = self.nn_module(input)
            if self.aux_weights is not None:
                loss = 0
                for pred, weight in zip(prediction, self.aux_weights):
                    loss += self.loss(pred, target, noisy) * weight
            else:
                loss = self.loss(prediction, target, noisy)
            return {
                'prediction': self.prediction_transform(prediction[0]),
                'target': target,
                'loss': loss.item(),
                'noisy': noisy
            }

    def predict(self, input):
        assert self.predict_ready()
        with torch.no_grad():
            if self.nn_module.training:
                self.nn_module.eval()
            input = deep_to(input, self.device)
            prediction = self.nn_module(input)
            if self.aux_weights is not None:
                prediction = prediction[0]
            prediction = self.prediction_transform(prediction)
            return prediction
