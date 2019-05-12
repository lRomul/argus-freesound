import torch
from torch.distributions.beta import Beta

from argus import Model
from argus.utils import deep_detach, deep_to

from src.models import resnet
from src.models import senet
from src.models.feature_extractor import FeatureExtractor
from src.models.simple_kaggle import SimpleKaggle
from src.losses import OnlyNoisyLqLoss, OnlyNoisyLSoftLoss, BCEMaxOutlierLoss, MixMatchLoss
from src import config


def sharpen(scores, temperature):
    scores_pow = scores.pow(1. / temperature)
    scores_pow_sum = scores_pow.sum(dim=1).reshape(-1, 1)
    return scores_pow / scores_pow_sum


def mixup(input1, target1, input2, target2, alpha):
    lambdas = Beta(alpha, alpha).sample(torch.Size((input1.shape[0],)))
    lambdas = lambdas.to(input1.device)
    lambdas = torch.max(lambdas, 1 - lambdas)

    input_lambdas = lambdas.view(-1, 1, 1, 1)
    input1 = input_lambdas * input1 + (1 - input_lambdas) * input2
    target_lambdas = lambdas.view(-1, 1)
    target1 = target_lambdas * target1 + (1 - target_lambdas) * target2

    return input1, target1


class MixMatchModel(Model):
    nn_module = {
        'resnet18': resnet.resnet18,
        'resnet34': resnet.resnet34,
        'FeatureExtractor': FeatureExtractor,
        'SimpleKaggle': SimpleKaggle,
        'se_resnext50_32x4d': senet.se_resnext50_32x4d
    }
    loss = {
        'OnlyNoisyLqLoss': OnlyNoisyLqLoss,
        'OnlyNoisyLSoftLoss': OnlyNoisyLSoftLoss,
        'BCEMaxOutlierLoss': BCEMaxOutlierLoss,
        'MixMatchLoss': MixMatchLoss
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params):
        super().__init__(params)
        self.T = params['mixmatch']['T']
        self.alpha = params['mixmatch']['alpha']
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

    def prepare_batch(self, batch, device):
        curated_input, curated_target, noisy_input = batch
        curated_input = deep_to(curated_input, device, non_blocking=True)
        curated_target = deep_to(curated_target, device, non_blocking=True)
        noisy_input = deep_to(noisy_input, device, non_blocking=True)
        return curated_input, curated_target, noisy_input

    def train_step(self, batch)-> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()
        curated_input, curated_target, noisy_input = self.prepare_batch(batch, self.device)

        with torch.no_grad():
            shape = noisy_input.shape
            noisy_input = noisy_input.view(shape[0] * shape[1], 1, shape[2], shape[3])
            gues_noisy_target = self.predict(noisy_input)
            self.nn_module.train()
            gues_noisy_target = gues_noisy_target.view(shape[0], shape[1], -1)
            gues_noisy_target = gues_noisy_target.mean(dim=1)
            gues_noisy_target = sharpen(gues_noisy_target, self.T)

            gues_noisy_target = torch.stack([gues_noisy_target] * shape[1], dim=1)
            gues_noisy_target = gues_noisy_target.view(shape[0] * shape[1], -1)

            w_input = torch.cat([curated_input, noisy_input], dim=0)
            w_target = torch.cat([curated_target, gues_noisy_target], dim=0)
            w_randperm = torch.randperm(w_input.shape[0])
            w_input = w_input[w_randperm]
            w_target = w_target[w_randperm]

            curated_input, curated_target = mixup(
                curated_input,
                curated_target,
                w_input[:curated_input.shape[0]],
                w_target[:curated_input.shape[0]],
                self.alpha
            )

            noisy_input, gues_noisy_target = mixup(
                noisy_input,
                gues_noisy_target,
                w_input[curated_input.shape[0]:],
                w_target[curated_input.shape[0]:],
                self.alpha
            )

            input = torch.cat([curated_input, noisy_input], dim=0)
            target = torch.cat([curated_target, gues_noisy_target], dim=0)
            noisy = torch.cat([
                torch.zeros(curated_input.shape[0], dtype=torch.uint8),
                torch.ones(noisy_input.shape[0], dtype=torch.uint8)
            ])
            noisy = noisy.to(self.device)

        prediction = self.nn_module(input)
        noisy_loss, curated_loss = self.loss(prediction, target, noisy)
        loss = noisy_loss + curated_loss
        if self.use_amp:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item(),
            'noisy_loss': noisy_loss.item(),
            'curated_loss': curated_loss.item()
        }

    def val_step(self, batch) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()
        with torch.no_grad():
            input, target, noisy = self.prepare_batch(batch, self.device)
            prediction = self.nn_module(input)
            noisy_loss, curated_loss = self.loss(prediction, target, noisy)
            loss = noisy_loss + curated_loss
            return {
                'prediction': self.prediction_transform(prediction),
                'target': target,
                'loss': loss.item(),
                'noisy_loss': noisy_loss.item(),
                'curated_loss': curated_loss.item()
            }
