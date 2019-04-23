import torch
import pandas as pd
import multiprocessing as mp
from torch.utils.data import Dataset

from src.audio import read_as_melspectrogram
from src import config


N_WORKERS = mp.cpu_count()


def get_folds_data():
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


class FreesoundDataset(Dataset):
    def __init__(self, folds_data, folds, transform=None):
        super().__init__()
        self.folds = folds
        self.transform = transform

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

        return image, target
