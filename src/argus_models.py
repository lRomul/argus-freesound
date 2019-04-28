import torch
import numpy as np
from argus import Model
from argus.utils import deep_to, deep_detach

from src.models.resnet import resnet18, resnet34
from src.models.feature_extractor import FeatureExtractor
from src.models.simple_kaggle import SimpleKaggle
from src.losses import sigmoid_mse_loss
from src.utils import sigmoid_rampup


class FreesoundModel(Model):
    nn_module = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'FeatureExtractor': FeatureExtractor,
        'SimpleKaggle': SimpleKaggle
    }
    prediction_transform = torch.nn.Sigmoid


class MeanTeacherFreesoundModel(Model):
    nn_module = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'FeatureExtractor': FeatureExtractor,
        'SimpleKaggle': SimpleKaggle
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params):
        super().__init__(params)
        self.alpha = params['mean_teacher']['alpha']
        self.rampup_length = params['mean_teacher']['rampup_length']
        self.unlabeled_batch = params['mean_teacher']['unlabeled_batch']
        self.unlabelled_dataset = None
        self.consistency_loss = sigmoid_mse_loss
        self.epoch = 0
        self.teacher = self._build_nn_module(self.params)
        self.teacher.to(self.device)
        self.teacher.train()
        for param in self.teacher.parameters():
            param.detach_()

    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(),
                                    self.nn_module.parameters()):
            t_param.data.mul_(self.alpha).add_(1 - self.alpha, s_param.data)

    def sample_unlabeled_input(self):
        assert self.unlabelled_dataset is not None
        indices = np.random.randint(0, len(self.unlabelled_dataset), size=self.unlabeled_batch)
        samples = [self.unlabelled_dataset[idx] for idx in indices]
        return torch.stack(samples, dim=0)

    def prepare_unlabeled_batch(self, batch, device):
        input, target = batch
        unlabeled_input = self.sample_unlabeled_input()
        input = torch.cat([input, unlabeled_input], dim=0)
        input = deep_to(input, device, non_blocking=True)
        target = deep_to(target, device, non_blocking=True)
        return input, target

    def train_step(self, batch) -> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()

        input, target = self.prepare_unlabeled_batch(batch, self.device)
        student_pred = self.nn_module(input)
        with torch.no_grad():
            teacher_pred = self.teacher(input)

        consistency_weight = sigmoid_rampup(self.epoch, self.rampup_length)
        loss = consistency_weight * self.consistency_loss(student_pred, teacher_pred)
        student_pred = student_pred[:target.size(0)]

        loss += self.loss(student_pred, target)
        loss.backward()
        self.optimizer.step()
        self.update_teacher()
        prediction = deep_detach(student_pred)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item()
        }
