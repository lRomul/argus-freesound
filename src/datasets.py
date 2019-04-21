import torch
import pandas as pd
import multiprocessing as mp
from torch.utils.data import Dataset

from src.audio import read_as_melspectrogram
from src import config


N_WORKERS = mp.cpu_count()


def get_samples(folds):
    train_folds_df = pd.read_csv(config.train_folds_path)

    audio_paths_lst = []
    targets_lst = []
    for i, row in train_folds_df.iterrows():
        if row.fold not in folds:
            continue

        audio_paths_lst.append(row.file_path)
        target = torch.zeros(len(config.classes))
        for label in row.labels.split(','):
            target[config.class2index[label]] = 1.
        targets_lst.append(target)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

    return images_lst, targets_lst


class FreesoundDataset(Dataset):
    def __init__(self, folds, transform=None):
        super().__init__()
        self.folds = folds
        self.transform = transform

        self.images_lst, self.targets_lst = get_samples(folds)

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        image = self.images_lst[idx].copy()
        target = self.targets_lst[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target
