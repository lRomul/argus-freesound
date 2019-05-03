import torch
import numpy as np
import random
import json
import time
from pprint import pprint

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import FreesoundDataset, CombinedDataset, FreesoundNoisyDataset
from src.transforms import get_transforms
from src.argus_models import FreesoundModel
from src.utils import load_folds_data, load_noisy_data
from src import config


EXPERIMENT_NAME = 'noisy_lsoft_rs_002'
VAL_FOLDS = [0]
TRAIN_FOLDS = [1, 2, 3, 4]
BATCH_SIZE = 128
CROP_SIZE = 128
DATASET_SIZE = 128 * 256
if config.kernel:
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 8
SAVE_DIR = config.experiments_dir / EXPERIMENT_NAME
START_FROM = 0


def train_experiment(folds_data, noisy_data, num):
    experiment_dir = SAVE_DIR / f'{num:04}'
    np.random.seed(num)
    random.seed(num)

    random_params = {
        'p_dropout': float(np.random.uniform(0.1, 0.3)),
        'batch_size': int(np.random.choice([128])),
        'lr': float(np.random.choice([0.001, 0.0006, 0.0003])),
        'add_prob': float(np.random.uniform(0.0, 1.0)),
        'noisy_prob': float(np.random.uniform(0.0, 1.0)),
        'lsoft_beta': float(np.random.uniform(0.2, 0.8)),
        'noisy_weight': float(np.random.uniform(0.3, 0.7)),
        'patience': int(np.random.randint(2, 10)),
        'factor': float(np.random.uniform(0.5, 0.8))
    }
    pprint(random_params)

    params = {
        'nn_module': ('SimpleKaggle', {
            'num_classes': len(config.classes),
            'dropout': random_params['p_dropout'],
            'base_size': 64
        }),
        'loss': ('OnlyNoisyLSoftLoss', {
            'beta': random_params['lsoft_beta'],
            'noisy_weight': random_params['noisy_weight'],
            'curated_weight': 1 - random_params['noisy_weight']
        }),
        'optimizer': ('Adam', {'lr': random_params['lr']}),
        'device': 'cuda',
        'amp': {
            'opt_level': 'O2',
            'keep_batchnorm_fp32': True,
            'loss_scale': "dynamic"
        }
    }
    pprint(params)
    try:
        train_transfrom = get_transforms(True, CROP_SIZE)
        curated_dataset = FreesoundDataset(folds_data, TRAIN_FOLDS,
                                           transform=train_transfrom,
                                           add_prob=random_params['add_prob'])
        noisy_dataset = FreesoundNoisyDataset(noisy_data,
                                              transform=train_transfrom)
        train_dataset = CombinedDataset(noisy_dataset, curated_dataset,
                                        noisy_prob=random_params['noisy_prob'],
                                        size=DATASET_SIZE)

        val_dataset = FreesoundDataset(folds_data, VAL_FOLDS,
                                       get_transforms(False, CROP_SIZE))
        train_loader = DataLoader(train_dataset, batch_size=random_params['batch_size'],
                                  shuffle=True, drop_last=True,
                                  num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=random_params['batch_size'] * 2,
                                shuffle=False, num_workers=NUM_WORKERS)

        model = FreesoundModel(params)

        callbacks = [
            MonitorCheckpoint(experiment_dir, monitor='val_lwlrap', max_saves=1),
            ReduceLROnPlateau(monitor='val_lwlrap',
                              patience=random_params['patience'],
                              factor=random_params['factor'],
                              min_lr=1e-8),
            EarlyStopping(monitor='val_lwlrap', patience=20),
            LoggingToFile(experiment_dir / 'log.txt'),
        ]

        with open(experiment_dir / 'random_params.json', 'w') as outfile:
            json.dump(random_params, outfile)

        model.fit(train_loader,
                  val_loader=val_loader,
                  max_epochs=100,
                  callbacks=callbacks,
                  metrics=['multi_accuracy', 'lwlrap'])
    except KeyboardInterrupt as e:
        raise e
    except BaseException as e:
        print(f"Exception '{e}' with random params '{random_params}'")


if __name__ == "__main__":
    print("Start load train data")
    noisy_data = load_noisy_data()
    folds_data = load_folds_data()

    for i in range(START_FROM, 10000):
        train_experiment(folds_data, noisy_data, i)
        time.sleep(5.0)
        torch.cuda.empty_cache()
        time.sleep(5.0)
