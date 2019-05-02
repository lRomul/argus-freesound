import json
import argparse

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import FreesoundDataset, RandomAddDataset
from src.datasets import CombinedDataset, FreesoundNoisyDataset
from src.transforms import get_transforms
from src.argus_models import FreesoundModel
from src.utils import load_folds_data, load_noisy_data
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

BATCH_SIZE = 128
CROP_SIZE = 128
DATASET_SIZE = 128 * 256
NOISY_PROB = 0.33
ADD_PROB = 0.5
if config.kernel:
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 8
SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('SimpleKaggle', {
        'num_classes': len(config.classes),
        'base_size': 64,
        'dropout': 0.111
    }),
    'loss': ('OnlyNoisyLSoftLoss', {
        'beta': 0.5,
        'noisy_weight': 0.5,
        'curated_weight': 0.5
    }),
    'optimizer': ('Adam', {'lr': 0.0006}),
    'device': 'cuda',
    'amp': {
        'opt_level': 'O2',
        'keep_batchnorm_fp32': True,
        'loss_scale': "dynamic"
    }
}


def train_fold(save_dir, train_folds, val_folds, folds_data, noisy_data):
    train_transfrom = get_transforms(True, CROP_SIZE)
    curated_dataset = RandomAddDataset(folds_data, train_folds,
                                       transform=train_transfrom,
                                       max_alpha=0.5, prob=ADD_PROB,
                                       min_add_target=0,
                                       max_add_target=1)
    noisy_dataset = FreesoundNoisyDataset(noisy_data,
                                          transform=train_transfrom)
    train_dataset = CombinedDataset(noisy_dataset, curated_dataset,
                                    noisy_prob=NOISY_PROB, size=DATASET_SIZE)

    val_dataset = FreesoundDataset(folds_data, val_folds,
                                   get_transforms(False, CROP_SIZE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = FreesoundModel(PARAMS)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_lwlrap', max_saves=1),
        ReduceLROnPlateau(monitor='val_lwlrap', patience=6, factor=0.6, min_lr=1e-8),
        EarlyStopping(monitor='val_lwlrap', patience=18),
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

    print("Model params", PARAMS)
    with open(SAVE_DIR / 'params.json', 'w') as outfile:
        json.dump(PARAMS, outfile)

    noisy_data = load_noisy_data()
    folds_data = load_folds_data()

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds, folds_data, noisy_data)
