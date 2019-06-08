import json
import time
import torch
import random
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
from torch.utils.data import Dataset

from src.audio import read_as_melspectrogram, get_audio_config
from src import config


N_WORKERS = mp.cpu_count()


def get_test_data():
    print("Start load test data")
    fname_lst = []
    wav_path_lst = []
    for wav_path in sorted(config.test_dir.glob('*.wav')):
        wav_path_lst.append(wav_path)
        fname_lst.append(wav_path.name)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, wav_path_lst)

    return fname_lst, images_lst


def get_folds_data(corrections=None):
    print("Start generate folds data")
    print("Audio config", get_audio_config())
    train_folds_df = pd.read_csv(config.train_folds_path)

    audio_paths_lst = []
    targets_lst = []
    folds_lst = []
    for i, row in train_folds_df.iterrows():
        labels = row.labels

        if corrections is not None:
            if row.fname in corrections:
                action = corrections[row.fname]
                if action == 'remove':
                    print(f"Skip {row.fname}")
                    continue
                else:
                    print(f"Replace labels {row.fname} from {labels} to {action}")
                    labels = action

        folds_lst.append(row.fold)
        audio_paths_lst.append(row.file_path)
        target = torch.zeros(len(config.classes))
        for label in labels.split(','):
            target[config.class2index[label]] = 1.
        targets_lst.append(target)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

    return images_lst, targets_lst, folds_lst


def get_augment_folds_data_generator(time_stretch_lst, pitch_shift_lst):
    print("Start generate augment folds data")
    print("Audio config", get_audio_config())
    print("time_stretch_lst:", time_stretch_lst)
    print("pitch_shift_lst:", pitch_shift_lst)
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

    yield images_lst, targets_lst, folds_lst
    images_lst = []

    for pitch_shift in pitch_shift_lst:
        pitch_shift_read = partial(read_as_melspectrogram, pitch_shift=pitch_shift)
        with mp.Pool(N_WORKERS) as pool:
            images_lst = pool.map(pitch_shift_read, audio_paths_lst)

        yield images_lst, targets_lst, folds_lst
        images_lst = []

    for time_stretch in time_stretch_lst:
        time_stretch_read = partial(read_as_melspectrogram, time_stretch=time_stretch)
        with mp.Pool(N_WORKERS) as pool:
            images_lst = pool.map(time_stretch_read, audio_paths_lst)

        yield images_lst, targets_lst, folds_lst
        images_lst = []


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


def get_noisy_data_generator():
    print("Start generate noisy data")
    print("Audio config", get_audio_config())
    train_noisy_df = pd.read_csv(config.train_noisy_csv_path)

    with open(config.noisy_corrections_json_path) as file:
        corrections = json.load(file)

    audio_paths_lst = []
    targets_lst = []
    for i, row in train_noisy_df.iterrows():
        labels = row.labels

        if row.fname in corrections:
            action = corrections[row.fname]
            if action == 'remove':
                continue
            else:
                labels = action

        audio_paths_lst.append(config.train_noisy_dir / row.fname)
        target = torch.zeros(len(config.classes))
        for label in labels.split(','):
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


class RandomDataset(Dataset):
    def __init__(self, datasets, p=None, size=4096):
        self.datasets = datasets
        self.p = p
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seed = int(time.time() * 1000.0) + idx
        random.seed(seed)
        np.random.seed(seed % (2**31))

        dataset_idx = np.random.choice(
            range(len(self.datasets)), p=self.p)
        dataset = self.datasets[dataset_idx]
        idx = random.randint(0, len(dataset) - 1)
        return dataset[idx]


def get_corrected_noisy_data():
    print("Start generate corrected noisy data")
    print("Audio config", get_audio_config())
    train_noisy_df = pd.read_csv(config.train_noisy_csv_path)

    with open(config.noisy_corrections_json_path) as file:
        corrections = json.load(file)

    audio_paths_lst = []
    targets_lst = []
    for i, row in train_noisy_df.iterrows():
        labels = row.labels

        if row.fname in corrections:
            action = corrections[row.fname]
            if action == 'remove':
                continue
            else:
                labels = action
        else:
            continue

        audio_paths_lst.append(config.train_noisy_dir / row.fname)
        target = torch.zeros(len(config.classes))

        for label in labels.split(','):
            target[config.class2index[label]] = 1.
        targets_lst.append(target)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

    return images_lst, targets_lst


class FreesoundCorrectedNoisyDataset(Dataset):
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

        noisy = torch.tensor(0, dtype=torch.uint8)
        return image, target, noisy
