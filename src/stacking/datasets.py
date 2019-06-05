import time
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from src import config


def load_fname_probs(experiments, fold, fname):
    prob_lst = []
    for experiment in experiments:
        npy_path = config.predictions_dir / experiment / f'fold_{fold}' / 'val' / (fname + '.npy')
        prob = np.load(npy_path)
        prob_lst.append(prob)
    probs = np.concatenate(prob_lst, axis=1)
    return probs.astype(np.float32)


def get_out_of_folds_data(experiments, corrections=None):
    train_folds_df = pd.read_csv(config.train_folds_path)

    probs_lst = []
    targets_lst = []
    folds_lst = []
    fname_lst = []
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
        probs = load_fname_probs(experiments, row.fold, row.fname)
        probs_lst.append(probs)
        target = torch.zeros(len(config.classes))
        for label in labels.split(','):
            target[config.class2index[label]] = 1.
        targets_lst.append(target)
        fname_lst.append(row.fname)

    return probs_lst, targets_lst, folds_lst, fname_lst


class StackingDataset(Dataset):
    def __init__(self, folds_data, folds,
                 transform=None, size=None,
                 spec_meta_info=None):
        super().__init__()
        self.folds = folds
        self.transform = transform
        self.size = size
        self.spec_meta_info = spec_meta_info

        self.probs_lst = []
        self.targets_lst = []
        self.fname_lst = []
        for prob, trg, fold, fname in zip(*folds_data):
            if fold in folds:
                self.probs_lst.append(prob)
                self.targets_lst.append(trg)
                self.fname_lst.append(fname)

    def __len__(self):
        if self.size is None:
            return len(self.probs_lst)
        else:
            return self.size

    def __getitem__(self, idx):
        if self.size is not None:
            seed = int(time.time() * 1000.0) + idx
            np.random.seed(seed % (2 ** 31))
            idx = np.random.randint(len(self.probs_lst))

        probs = self.probs_lst[idx].copy()
        target = self.targets_lst[idx].clone()

        if self.transform is not None:
            probs = self.transform(probs)

        if self.spec_meta_info is not None:
            fname = self.fname_lst[idx]
            spec_meta = self.spec_meta_info[fname]
            spec_meta = torch.from_numpy(spec_meta.astype(np.float32))
            probs = torch.cat([probs, spec_meta], dim=0)

        return probs, target
