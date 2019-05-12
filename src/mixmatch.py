import time
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class EmptyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


class MixMatchCollate:
    def __init__(self,
                 curated_dataset,
                 noisy_dataset,
                 K,
                 transform):
        self.curated_dataset = curated_dataset
        self.noisy_dataset = noisy_dataset
        self.K = K
        self.transform = transform

    def __call__(self, indices):
        seed = int(time.time() * 1000.0) + indices[0] + + indices[-1]
        random.seed(seed)

        curated_tensors = []
        curated_targets = []
        for _ in indices:
            ind = random.randint(0, len(self.curated_dataset) - 1)
            image = self.curated_dataset.images_lst[ind].copy()
            target = self.curated_dataset.targets_lst[ind].clone()
            tensor = self.transform(image)

            curated_tensors.append(tensor)
            curated_targets.append(target)

        curated_tensor = torch.stack(curated_tensors, dim=0)
        curated_target = torch.stack(curated_targets, dim=0)

        noisy_tensors = []

        for _ in indices:
            ind = random.randint(0, len(self.noisy_dataset) - 1)
            image = self.noisy_dataset.images_lst[ind].copy()
            augm_tensors = []
            for k in range(self.K):
                tensor = self.transform(image)
                augm_tensors.append(tensor)

            augm_tensor = torch.cat(augm_tensors, dim=0)
            noisy_tensors.append(augm_tensor)

        noisy_tensor = torch.stack(noisy_tensors, dim=0)

        return curated_tensor, curated_target, noisy_tensor
