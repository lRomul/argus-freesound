import re
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.stats.mstats import gmean

from src.datasets import get_noisy_data_generator, get_folds_data, get_augment_folds_data_generator
from src import config


def gmean_preds_blend(probs_df_lst):
    blend_df = probs_df_lst[0]
    blend_values = np.stack([df.loc[blend_df.index.values].values
                             for df in probs_df_lst], axis=0)
    blend_values = gmean(blend_values, axis=0)

    blend_df.values[:] = blend_values
    return blend_df


def get_best_model_path(dir_path: Path, return_score=False):
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = float(score.group(0)[1:-4])
            model_scores.append((model_path, score))
    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_path = model_score[-1][0]
    if return_score:
        best_score = model_score[-1][1]
        return best_model_path, best_score
    else:
        return best_model_path


def pickle_save(obj, filename):
    print(f"Pickle save to: {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    print(f"Pickle load from: {filename}")
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def load_folds_data(use_corrections=True):
    if use_corrections:
        with open(config.corrections_json_path) as file:
            corrections = json.load(file)
        print("Corrections:", corrections)
        pkl_name = f'{config.audio.get_hash(corrections=corrections)}.pkl'
    else:
        corrections = None
        pkl_name = f'{config.audio.get_hash()}.pkl'

    folds_data_pkl_path = config.folds_data_pkl_dir / pkl_name

    if folds_data_pkl_path.exists():
        folds_data = pickle_load(folds_data_pkl_path)
    else:
        folds_data = get_folds_data(corrections)
        if not config.folds_data_pkl_dir.exists():
            config.folds_data_pkl_dir.mkdir(parents=True, exist_ok=True)
        pickle_save(folds_data, folds_data_pkl_path)
    return folds_data


def load_noisy_data():
    with open(config.noisy_corrections_json_path) as file:
        corrections = json.load(file)

    pkl_name_glob = f'{config.audio.get_hash(corrections=corrections)}_*.pkl'
    pkl_paths = sorted(config.noisy_data_pkl_dir.glob(pkl_name_glob))

    images_lst, targets_lst = [], []

    if pkl_paths:
        for pkl_path in pkl_paths:
            data_batch = pickle_load(pkl_path)
            images_lst += data_batch[0]
            targets_lst += data_batch[1]
    else:
        if not config.noisy_data_pkl_dir.exists():
            config.noisy_data_pkl_dir.mkdir(parents=True, exist_ok=True)

        for i, data_batch in enumerate(get_noisy_data_generator()):
            pkl_name = f'{config.audio.get_hash(corrections=corrections)}_{i:02}.pkl'
            noisy_data_pkl_path = config.noisy_data_pkl_dir / pkl_name
            pickle_save(data_batch, noisy_data_pkl_path)

            images_lst += data_batch[0]
            targets_lst += data_batch[1]

    return images_lst, targets_lst


def load_augment_folds_data(time_stretch_lst, pitch_shift_lst):
    config_hash = config.audio.get_hash(time_stretch_lst=time_stretch_lst,
                                        pitch_shift_lst=pitch_shift_lst)
    pkl_name_glob = f'{config_hash}_*.pkl'
    pkl_paths = sorted(config.augment_folds_data_pkl_dir.glob(pkl_name_glob))

    images_lst, targets_lst, folds_lst = [], [], []

    if pkl_paths:
        for pkl_path in pkl_paths:
            data_batch = pickle_load(pkl_path)
            images_lst += data_batch[0]
            targets_lst += data_batch[1]
            folds_lst += data_batch[2]
    else:
        if not config.augment_folds_data_pkl_dir.exists():
            config.augment_folds_data_pkl_dir.mkdir(parents=True, exist_ok=True)

        generator = get_augment_folds_data_generator(time_stretch_lst, pitch_shift_lst)
        for i, data_batch in enumerate(generator):
            pkl_name = f'{config_hash}_{i:02}.pkl'
            augment_data_pkl_path = config.augment_folds_data_pkl_dir / pkl_name
            pickle_save(data_batch, augment_data_pkl_path)

            images_lst += data_batch[0]
            targets_lst += data_batch[1]
            folds_lst += data_batch[2]

    return images_lst, targets_lst, folds_lst
