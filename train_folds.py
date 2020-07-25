import json
import argparse

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import FreesoundDataset, FreesoundNoisyDataset, RandomDataset
from src.datasets import get_corrected_noisy_data, FreesoundCorrectedNoisyDataset
from src.mixers import RandomMixer, AddMixer, SigmoidConcatMixer, UseMixerWithProb
from src.transforms import get_transforms
from src.argus_models import FreesoundModel
from src.utils import load_noisy_data, load_folds_data
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()

BATCH_SIZE = 128
CROP_SIZE = 256
DATASET_SIZE = 128 * 256
NOISY_PROB = 0.01
CORR_NOISY_PROB = 0.42
MIXER_PROB = 0.8
WRAP_PAD_PROB = 0.5
CORRECTIONS = True
if config.kernel:
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 8
SAVE_DIR = config.experiments_dir / args.experiment
PARAMS = {
    'nn_module': ('AuxSkipAttention', {
        'num_classes': len(config.classes),
        'base_size': 64,
        'dropout': 0.4,
        'ratio': 16,
        'kernel_size': 7,
        'last_filters': 8,
        'last_fc': 4
    }),
    'loss': ('OnlyNoisyLSoftLoss', {
        'beta': 0.7,
        'noisy_weight': 0.5,
        'curated_weight': 0.5
    }),
    'optimizer': ('Adam', {'lr': 0.0009}),
    'device': 'cuda',
    'aux': {
        'weights': [1.0, 0.4, 0.2, 0.1]
    },
    'amp': {
        'opt_level': 'O2',
        'keep_batchnorm_fp32': True,
        'loss_scale': "dynamic"
    }
}


def train_fold(save_dir, train_folds, val_folds,
               folds_data, noisy_data, corrected_noisy_data):
    train_transfrom = get_transforms(train=True,
                                     size=CROP_SIZE,
                                     wrap_pad_prob=WRAP_PAD_PROB,
                                     resize_scale=(0.8, 1.0),
                                     resize_ratio=(1.7, 2.3),
                                     resize_prob=0.33,
                                     spec_num_mask=2,
                                     spec_freq_masking=0.15,
                                     spec_time_masking=0.20,
                                     spec_prob=0.5)

    mixer = RandomMixer([
        SigmoidConcatMixer(sigmoid_range=(3, 12)),
        AddMixer(alpha_dist='uniform')
    ], p=[0.6, 0.4])
    mixer = UseMixerWithProb(mixer, prob=MIXER_PROB)

    curated_dataset = FreesoundDataset(folds_data, train_folds,
                                       transform=train_transfrom,
                                       mixer=mixer)
    noisy_dataset = FreesoundNoisyDataset(noisy_data,
                                          transform=train_transfrom,
                                          mixer=mixer)
    corr_noisy_dataset = FreesoundCorrectedNoisyDataset(corrected_noisy_data,
                                                        transform=train_transfrom,
                                                        mixer=mixer)
    dataset_probs = [NOISY_PROB, CORR_NOISY_PROB, 1 - NOISY_PROB - CORR_NOISY_PROB]
    print("Dataset probs", dataset_probs)
    print("Dataset lens", len(noisy_dataset), len(corr_noisy_dataset), len(curated_dataset))
    train_dataset = RandomDataset([noisy_dataset, corr_noisy_dataset, curated_dataset],
                                  p=dataset_probs,
                                  size=DATASET_SIZE)

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
              num_epochs=700,
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

    folds_data = load_folds_data(use_corrections=CORRECTIONS)
    noisy_data = load_noisy_data()
    corrected_noisy_data = get_corrected_noisy_data()

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds,
                   folds_data, noisy_data, corrected_noisy_data)
