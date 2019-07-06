import json
import numpy as np
import pandas as pd

from src.stacking.datasets import load_fname_probs
from src.stacking.predictor import StackPredictor
from src.metrics import LwlrapBase
from src.utils import get_best_model_path
from src import config


STACKING_EXPERIMENT = "stacking_008_fcnet_50013"

EXPERIMENTS = [
    'auxiliary_016',
    'auxiliary_019',
    'corr_noisy_003',
    'corr_noisy_004',
    'corr_noisy_007',
    'corrections_002',
    'corrections_003'
]

EXPERIMENT_DIR = config.experiments_dir / STACKING_EXPERIMENT
PREDICTION_DIR = config.predictions_dir / STACKING_EXPERIMENT
DEVICE = 'cuda'
BATCH_SIZE = 256


def pred_val_fold(predictor, fold):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'val'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    train_folds_df = pd.read_csv(config.train_folds_path)
    train_folds_df = train_folds_df[train_folds_df.fold == fold]

    fname_lst = []
    probs_lst = []
    for i, row in train_folds_df.iterrows():
        probs = load_fname_probs(EXPERIMENTS, fold, row.fname)

        probs_lst.append(probs.mean(axis=0))
        fname_lst.append(row.fname)

    stack_probs = np.stack(probs_lst, axis=0)
    preds = predictor.predict(stack_probs)

    probs_df = pd.DataFrame(data=list(preds),
                            index=fname_lst,
                            columns=config.classes)
    probs_df.index.name = 'fname'
    probs_df.to_csv(fold_prediction_dir / 'probs.csv')


def calc_lwlrap_on_val():
    probs_df_lst = []
    for fold in config.folds:
        fold_probs_path = PREDICTION_DIR / f'fold_{fold}' / 'val' / 'probs.csv'
        probs_df = pd.read_csv(fold_probs_path)
        probs_df.set_index('fname', inplace=True)
        probs_df_lst.append(probs_df)

    probs_df = pd.concat(probs_df_lst, axis=0)
    train_curated_df = pd.read_csv(config.train_curated_csv_path)

    lwlrap = LwlrapBase(config.classes)
    for i, row in train_curated_df.iterrows():
        target = np.zeros(len(config.classes))
        for label in row.labels.split(','):
            target[config.class2index[label]] = 1.

        pred = probs_df.loc[row.fname].values
        lwlrap.accumulate(target[np.newaxis], pred[np.newaxis])

    result = {
        'overall_lwlrap': lwlrap.overall_lwlrap(),
        'per_class_lwlrap': {cls: lwl for cls, lwl in zip(config.classes,
                                                          lwlrap.per_class_lwlrap())}
    }
    print(result)
    with open(PREDICTION_DIR / 'val_lwlrap.json', 'w') as file:
        json.dump(result, file, indent=2)


if __name__ == "__main__":
    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = EXPERIMENT_DIR / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = StackPredictor(model_path,
                                   BATCH_SIZE,
                                   device=DEVICE)

        print("Val predict")
        pred_val_fold(predictor, fold)

    print("Calculate lwlrap metric on cv")
    calc_lwlrap_on_val()
