import pickle

from src.datasets import get_noisy_data_generator, get_folds_data
from src import config


def pickle_save(obj, filename):
    print(f"Pickle save to: {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    print(f"Pickle load from: {filename}")
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def load_folds_data():
    pkl_name = f'{config.audio.get_hash()}.pkl'
    folds_data_pkl_path = config.folds_data_pkl_dir / pkl_name

    if folds_data_pkl_path.exists():
        folds_data = pickle_load(folds_data_pkl_path)
    else:
        folds_data = get_folds_data()
        if not config.folds_data_pkl_dir.exists():
            config.folds_data_pkl_dir.mkdir(parents=True, exist_ok=True)
        pickle_save(folds_data, folds_data_pkl_path)
    return folds_data


def load_noisy_data():
    pkl_name_glob = f'{config.audio.get_hash()}_*.pkl'
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
            pkl_name = f'{config.audio.get_hash()}_{i:02}.pkl'
            noisy_data_pkl_path = config.noisy_data_pkl_dir / pkl_name
            pickle_save(data_batch, noisy_data_pkl_path)

            images_lst += data_batch[0]
            targets_lst += data_batch[1]

    return images_lst, targets_lst