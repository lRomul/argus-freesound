import json
import time
import torch
import random
import numpy as np
from pprint import pprint

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.stacking.datasets import get_out_of_folds_data, StackingDataset
from src.stacking.transforms import get_transforms
from src.stacking.argus_models import StackingModel
from src import config

EXPERIMENT_NAME = 'fcnet_stacking_rs_004'
START_FROM = 0
EXPERIMENTS = [
    'auxiliary_007',
    'auxiliary_010',
    'auxiliary_012',
    'auxiliary_014'
]
DATASET_SIZE = 128 * 256
CORRECTIONS = True
if config.kernel:
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 4
SAVE_DIR = config.experiments_dir / EXPERIMENT_NAME


def train_folds(save_dir, folds_data):
    random_params = {
        'base_size': int(np.random.choice([64, 128, 256, 512])),
        'reduction_scale': int(np.random.choice([2, 4, 8, 16])),
        'p_dropout': float(np.random.uniform(0.0, 0.5)),
        'lr': float(np.random.uniform(0.0001, 0.00001)),
        'patience': int(np.random.randint(3, 12)),
        'factor': float(np.random.uniform(0.5, 0.8)),
        'batch_size': int(np.random.choice([32, 64, 128])),
    }
    pprint(random_params)

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'random_params.json', 'w') as outfile:
        json.dump(random_params, outfile)

    params = {
        'nn_module': ('FCNet', {
            'in_channels': len(config.classes) * len(EXPERIMENTS),
            'num_classes': len(config.classes),
            'base_size': random_params['base_size'],
            'reduction_scale': random_params['reduction_scale'],
            'p_dropout': random_params['p_dropout']
        }),
        'loss': 'BCEWithLogitsLoss',
        'optimizer': ('Adam', {'lr': random_params['lr']}),
        'device': 'cuda',
    }

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = save_dir / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")

        train_dataset = StackingDataset(folds_data, train_folds,
                                        get_transforms(True),
                                        DATASET_SIZE)
        val_dataset = StackingDataset(folds_data, val_folds,
                                      get_transforms(False))

        train_loader = DataLoader(train_dataset,
                                  batch_size=random_params['batch_size'],
                                  shuffle=True, drop_last=True,
                                  num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset,
                                batch_size=random_params['batch_size'] * 2,
                                shuffle=False, num_workers=NUM_WORKERS)

        model = StackingModel(params)

        callbacks = [
            MonitorCheckpoint(save_fold_dir, monitor='val_lwlrap', max_saves=1),
            ReduceLROnPlateau(monitor='val_lwlrap',
                              patience=random_params['patience'],
                              factor=random_params['factor'],
                              min_lr=1e-8),
            EarlyStopping(monitor='val_lwlrap', patience=20),
            LoggingToFile(save_fold_dir / 'log.txt'),
        ]

        model.fit(train_loader,
                  val_loader=val_loader,
                  max_epochs=300,
                  callbacks=callbacks,
                  metrics=['multi_accuracy', 'lwlrap'])


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    if CORRECTIONS:
        with open(config.corrections_json_path) as file:
            corrections = json.load(file)
        print("Corrections:", corrections)
    else:
        corrections = None

    folds_data = get_out_of_folds_data(EXPERIMENTS, corrections)

    for num in range(START_FROM, 10000):
        np.random.seed(num)
        random.seed(num)

        save_dir = SAVE_DIR / f'{num:04}'
        train_folds(save_dir, folds_data)
        time.sleep(5.0)
        torch.cuda.empty_cache()
        time.sleep(5.0)
