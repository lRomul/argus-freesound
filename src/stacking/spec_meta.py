import json
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from src.audio import read_as_melspectrogram
from src.utils import pickle_load, pickle_save
from src import config

N_WORKERS = mp.cpu_count()


def spec_to_meta(spec):
    duration = np.clip(spec.shape[1], 32.0, 2000.0)
    duration = np.log(duration / 32.0) / 4.0
    spec_stat = spec.mean(axis=1) / 100.0
    meta = np.concatenate(([duration], spec_stat), axis=0)
    return meta.astype(np.float32)


def load_spec_meta_info(use_corrections=True):
    if use_corrections:
        with open(config.corrections_json_path) as file:
            corrections = json.load(file)
        pkl_name = f'{config.audio.get_hash(corrections=corrections)}.pkl'
    else:
        corrections = None
        pkl_name = f'{config.audio.get_hash()}.pkl'

    spec_meta_info_dir_path = config.spec_meta_info_dir / pkl_name

    if spec_meta_info_dir_path.exists():
        spec_meta_info = pickle_load(spec_meta_info_dir_path)
    else:
        spec_meta_info = get_spec_meta_info(corrections)
        if not config.spec_meta_info_dir.exists():
            config.spec_meta_info_dir.mkdir(parents=True, exist_ok=True)
        pickle_save(spec_meta_info, spec_meta_info_dir_path)
    return spec_meta_info


def get_spec_meta_info(corrections=None):
    train_folds_df = pd.read_csv(config.train_folds_path)

    audio_paths_lst = []
    fname_lst = []
    for i, row in train_folds_df.iterrows():
        labels = row.labels

        if corrections is not None:
            if row.fname in corrections:
                action = corrections[row.fname]
                if action == 'remove':
                    continue
                else:
                    labels = action

        audio_paths_lst.append(row.file_path)
        target = torch.zeros(len(config.classes))
        for label in labels.split(','):
            target[config.class2index[label]] = 1.
        fname_lst.append(row.fname)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, audio_paths_lst)

    spec_meta_info = dict()
    for fname, image in zip(fname_lst, images_lst):
        spec_meta_info[fname] = spec_to_meta(image)

    return spec_meta_info
