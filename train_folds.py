import os
import json

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import FreesoundDataset
from src.transforms import get_transforms
from src.argus_models import CnnFinetune
from src import config


EXPERIMENT_NAME = 'test_001'
BATCH_SIZE = 32
CROP_SIZE = 128
SAVE_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
FOLDS = config.folds
PARAMS = {
    'nn_module': {
        'model_name': 'resnet34',
        'num_classes': len(config.classes),
        'pretrained': False,
        'dropout_p': 0.2
    },
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('Adam', {'lr': 0.001}),
    'device': 'cuda'
}


def train_fold(save_dir, train_folds, val_folds):
    train_dataset = FreesoundDataset(train_folds, get_transforms(True, CROP_SIZE))
    val_dataset = FreesoundDataset(val_folds, get_transforms(False, CROP_SIZE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

    model = CnnFinetune(PARAMS)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_multi_accuracy', max_saves=3),
        ReduceLROnPlateau(monitor='val_multi_accuracy', patience=20, factor=0.64, min_lr=1e-8),
        EarlyStopping(monitor='val_multi_accuracy', patience=50),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=150,
              callbacks=callbacks,
              metrics=['multi_accuracy'])


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(os.path.join(SAVE_DIR, 'source.py'), 'w') as outfile:
        outfile.write(open(__file__).read())

    with open(os.path.join(SAVE_DIR, 'params.json'), 'w') as outfile:
        json.dump(PARAMS, outfile)

    for i in range(len(FOLDS)):
        val_folds = [FOLDS[i]]
        train_folds = FOLDS[:i] + FOLDS[i + 1:]
        save_fold_dir = os.path.join(SAVE_DIR, f'fold_{FOLDS[i]}')
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds)
