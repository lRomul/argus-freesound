import time
import torch
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from torch.utils.data import Dataset

from src.audio import read_as_melspectrogram, get_audio_config
from src import config


N_WORKERS = mp.cpu_count()


def get_folds_data():
    print("Start generate folds data")
    print("Audio config", get_audio_config())
    train_folds_df = pd.read_csv(config.train_folds_path)

    audio_paths_lst = []
    targets_lst = []
    folds_lst = []
    for i, row in train_folds_df.iterrows():
        folds_lst.append(row.fold)
        audio_paths_lst.append(row.file_path)
        target = torch.zeros(len(config.classes))
        for label in row.labels.split(','):
            target[config.class2index[label]] = 1.
        targets_lst.append(target)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

    return images_lst, targets_lst, folds_lst


class RandomMixer:
    def __init__(self, mix_prob=0.5, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.mix_prob = mix_prob
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        if random.random() < self.mix_prob:
            rnd_idx = random.randint(0, len(dataset) - 1)
            rnd_image = dataset.images_lst[rnd_idx].copy()
            rnd_target = dataset.targets_lst[rnd_idx].clone()
            rnd_image = dataset.transform(rnd_image)
            alpha = self.sample_alpha()
            image = (1 - alpha) * image + alpha * rnd_image
            target = (1 - alpha) * target + alpha * rnd_target
        return image, target


class FreesoundDataset(Dataset):
    def __init__(self, folds_data, folds,
                 transform=None,
                 mixer=None):
        super().__init__()
        self.folds = folds
        self.transform = transform
        self.mixer = mixer

        self.images_lst = []
        self.targets_lst = []
        for img, trg, fold in zip(*folds_data):
            if fold in folds:
                self.images_lst.append(img)
                self.targets_lst.append(trg)

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        image = self.images_lst[idx].copy()
        target = self.targets_lst[idx].clone()

        if self.transform is not None:
            image = self.transform(image)

        if self.mixer is not None:
            image, target = self.mixer(self, image, target)

        noisy = torch.tensor(0, dtype=torch.uint8)
        return image, target, noisy


def get_noisy_data():
    print("Start generate noisy data")
    print("Audio config", get_audio_config())
    train_noisy_df = pd.read_csv(config.train_noisy_csv_path)

    audio_paths_lst = []
    targets_lst = []
    for i, row in train_noisy_df.iterrows():
        audio_paths_lst.append(config.train_noisy_dir / row.fname)
        target = torch.zeros(len(config.classes))
        for label in row.labels.split(','):
            target[config.class2index[label]] = 1.
        targets_lst.append(target)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

    return images_lst, targets_lst


def get_noisy_data_generator():
    print("Start generate noisy data")
    print("Audio config", get_audio_config())
    train_noisy_df = pd.read_csv(config.train_noisy_csv_path)

    audio_paths_lst = []
    targets_lst = []
    for i, row in train_noisy_df.iterrows():
        audio_paths_lst.append(config.train_noisy_dir / row.fname)
        target = torch.zeros(len(config.classes))
        for label in row.labels.split(','):
            target[config.class2index[label]] = 1.
        targets_lst.append(target)

        if len(audio_paths_lst) >= 5000:
            with mp.Pool(N_WORKERS) as pool:
                images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

            yield images_lst, targets_lst

            audio_paths_lst = []
            images_lst = []
            targets_lst = []

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

    yield images_lst, targets_lst


class FreesoundNoisyDataset(Dataset):
    def __init__(self, noisy_data, transform=None,
                 mixer=None):
        super().__init__()
        self.transform = transform
        self.mixer = mixer

        self.images_lst = []
        self.targets_lst = []
        for img, trg in zip(*noisy_data):
            self.images_lst.append(img)
            self.targets_lst.append(trg)

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        image = self.images_lst[idx].copy()
        target = self.targets_lst[idx].clone()

        if self.transform is not None:
            image = self.transform(image)

        if self.mixer is not None:
            image, target = self.mixer(self, image, target)

        noisy = torch.tensor(1, dtype=torch.uint8)
        return image, target, noisy


class CombinedDataset(Dataset):
    def __init__(self, noisy_dataset, curated_dataset,
                 noisy_prob=0.5, size=4096):
        self.noisy_dataset = noisy_dataset
        self.curated_dataset = curated_dataset
        self.noisy_prob = noisy_prob
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seed = int(time.time() * 1000.0) + idx
        random.seed(seed)
        np.random.seed(seed % (2**31))

        if random.random() < self.noisy_prob:
            idx = random.randint(0, len(self.noisy_dataset) - 1)
            return self.noisy_dataset[idx]

        else:
            idx = random.randint(0, len(self.curated_dataset) - 1)
            return self.curated_dataset[idx]
