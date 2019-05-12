import json
import argparse

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import FreesoundDataset, FreesoundNoisyDataset
from src.transforms import get_transforms
from src.mixmatch_argus_models import MixMatchModel
from src.mixmatch import EmptyDataset, MixMatchCollate
from src.mixers import SigmoidConcatMixer, UseMixerWithProb
from src.utils import load_noisy_data, load_folds_data
from src import config

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

BATCH_SIZE = 48
CROP_SIZE = 256
DATASET_SIZE = 128 * 128
K = 2
WRAP_PAD_PROB = 0.5
if config.kernel:
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 8
SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('SimpleKaggle', {
        'num_classes': len(config.classes),
        'in_channels': 1,
        'base_size': 64,
        'dropout': 0.111
    }),
    'loss': ('OnlyNoisyLSoftLoss', {
        'beta': 0.5,
        'noisy_weight': 0.5,
        'curated_weight': 1.0
    }),
    'optimizer': ('Adam', {'lr': 0.0006}),
    'device': 'cuda',
    'mixmatch': {
        'T': 0.5,
        'alpha': 0.75
    },
    'amp': {
        'opt_level': 'O2',
        'keep_batchnorm_fp32': True,
        'loss_scale': "dynamic"
    }
}


def train_fold(save_dir, train_folds, val_folds,
               folds_data, noisy_data):
    train_transfrom = get_transforms(train=True,
                                     size=CROP_SIZE,
                                     wrap_pad_prob=WRAP_PAD_PROB)
    mixer = UseMixerWithProb(
        SigmoidConcatMixer(sigmoid_range=(3, 12)),
        prob=0.5
    )
    curated_dataset = FreesoundDataset(folds_data, train_folds, mixer=mixer)
    noisy_dataset = FreesoundNoisyDataset(noisy_data, mixer=mixer)
    collate = MixMatchCollate(curated_dataset, noisy_dataset,
                              K, train_transfrom)

    train_dataset = EmptyDataset(size=DATASET_SIZE)
    val_dataset = FreesoundDataset(folds_data, val_folds,
                                   get_transforms(False, CROP_SIZE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = MixMatchModel(PARAMS)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_lwlrap', max_saves=1),
        ReduceLROnPlateau(monitor='val_lwlrap', patience=9, factor=0.6, min_lr=1e-8),
        EarlyStopping(monitor='val_lwlrap', patience=27),
        LoggingToFile(save_dir / 'log.txt'),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=700,
              callbacks=callbacks,
              metrics=['multi_accuracy', 'lwlrap', 'noisy_curated_loss'],
              metrics_on_train=True)


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

    folds_data = load_folds_data()
    noisy_data = load_noisy_data()

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds,
                   folds_data, noisy_data)
