from argus import load_model
from argus.callbacks import MonitorCheckpoint, LoggingToFile

from torch.utils.data import DataLoader

from src.datasets import FreesoundDataset, FreesoundNoisyDataset, RandomDataset
from src.mixers import RandomMixer, AddMixer, SigmoidConcatMixer, UseMixerWithProb
from src.transforms import get_transforms
from src.argus_models import FreesoundModel
from src.lr_scheduler import CosineAnnealing
from src.utils import load_noisy_data, load_folds_data, get_best_model_path
from src import config


BASE_EXPERIMENT_NAME = 'noisy_mixup_001'
EXPERIMENT_NAME = 'noisy_mixup_001_after_001'

BATCH_SIZE = 128
CROP_SIZE = 256
DATASET_SIZE = 128 * 256
NOISY_PROB = 0.33
MIXER_PROB = 0.66
WRAP_PAD_PROB = 0.5
BASE_LR = 0.0003
if config.kernel:
    NUM_WORKERS = 2
else:
    NUM_WORKERS = 8
DEVICE = 'cuda'
BASE_DIR = config.experiments_dir / BASE_EXPERIMENT_NAME
SAVE_DIR = config.experiments_dir / EXPERIMENT_NAME


def train_fold(base_model_path, save_dir, train_folds, val_folds,
               folds_data, noisy_data):
    train_transfrom = get_transforms(train=True,
                                     size=CROP_SIZE,
                                     wrap_pad_prob=WRAP_PAD_PROB)

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
    train_dataset = RandomDataset([noisy_dataset, curated_dataset],
                                  p=[NOISY_PROB, 1 - NOISY_PROB],
                                  size=DATASET_SIZE)

    val_dataset = FreesoundDataset(folds_data, val_folds,
                                   get_transforms(False, CROP_SIZE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = load_model(base_model_path, device=DEVICE)
    model.set_lr(BASE_LR)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_lwlrap', max_saves=3),
        CosineAnnealing(T_0=10, T_mult=2, eta_min=0.00001),
        LoggingToFile(save_dir / 'log.txt'),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              num_epochs=150,
              callbacks=callbacks,
              metrics=['multi_accuracy', 'lwlrap'])


if __name__ == "__main__":
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(SAVE_DIR / 'source.py', 'w') as outfile:
        outfile.write(open(__file__).read())

    folds_data = load_folds_data()
    noisy_data = load_noisy_data()

    for fold in config.folds:
        val_folds = [fold]
        train_folds = list(set(config.folds) - set(val_folds))
        save_fold_dir = SAVE_DIR / f'fold_{fold}'
        base_model_path = get_best_model_path(BASE_DIR / f'fold_{fold}')
        print(f"Base model path: {base_model_path}")
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(base_model_path, save_fold_dir, train_folds, val_folds,
                   folds_data, noisy_data)
