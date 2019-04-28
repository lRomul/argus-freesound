import json
import argparse

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import FreesoundDataset, get_folds_data
from src.datasets import FreesoundNoisyDataset, get_noisy_data
from src.transforms import get_transforms
from src.argus_models import MeanTeacherFreesoundModel
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

BATCH_SIZE = 64
UNLABELED_BATCH = 32
CROP_SIZE = 128
SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('SimpleKaggle', {
        'num_classes': len(config.classes),
        'base_size': 64,
        'dropout': 0.275
    }),
    'loss': 'BCEWithLogitsLoss',
    'optimizer': ('Adam', {'lr': 0.001}),
    'device': 'cuda',
    'mean_teacher': {
        'alpha': 0.99,
        'rampup_length': 100,
        'unlabeled_batch': UNLABELED_BATCH,
    }
}


def train_fold(save_dir, train_folds, val_folds, folds_data, noisy_data):
    train_transforms = get_transforms(True, CROP_SIZE)
    train_dataset = FreesoundDataset(folds_data, train_folds,
                                     train_transforms)
    val_dataset = FreesoundDataset(folds_data, val_folds,
                                   get_transforms(False, CROP_SIZE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    noisy_dataset = FreesoundNoisyDataset(noisy_data, train_transforms, False)

    model = MeanTeacherFreesoundModel(PARAMS)
    model.unlabelled_dataset = noisy_dataset

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_lwlrap', max_saves=1),
        ReduceLROnPlateau(monitor='val_lwlrap', patience=34, factor=0.536, min_lr=1e-8),
        EarlyStopping(monitor='val_lwlrap', patience=70),
        LoggingToFile(save_dir / 'log.txt'),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=700,
              callbacks=callbacks,
              metrics=['multi_accuracy', 'lwlrap'])


if __name__ == "__main__":
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    with open(SAVE_DIR / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    print("Start load train data")
    folds_data = get_folds_data()
    print("Start load noisy data")
    noisy_data = get_noisy_data()

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data, noisy_data)
