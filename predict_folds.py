import json
import argparse
import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.audio import read_as_melspectrogram
from src.transforms import get_transforms
from src.metrics import LwlrapBase
from src.utils import get_best_model_path, gmean_preds_blend
from src.datasets import get_test_data
from src import config


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True, type=str)
args = parser.parse_args()


EXPERIMENT_DIR = config.experiments_dir / args.experiment
PREDICTION_DIR = config.predictions_dir / args.experiment
DEVICE = 'cuda'
CROP_SIZE = 256
BATCH_SIZE = 16


def pred_val_fold(predictor, fold):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'val'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    train_folds_df = pd.read_csv(config.train_folds_path)
    train_folds_df = train_folds_df[train_folds_df.fold == fold]

    fname_lst = []
    pred_lst = []
    for i, row in train_folds_df.iterrows():
        image = read_as_melspectrogram(row.file_path)
        pred = predictor.predict(image)

        pred_path = fold_prediction_dir / f'{row.fname}.npy'
        np.save(pred_path, pred)

        pred = pred.mean(axis=0)
        pred_lst.append(pred)
        fname_lst.append(row.fname)

    preds = np.stack(pred_lst, axis=0)
    probs_df = pd.DataFrame(data=preds,
                            index=fname_lst,
                            columns=config.classes)
    probs_df.index.name = 'fname'
    probs_df.to_csv(fold_prediction_dir / 'probs.csv')


def pred_test_fold(predictor, fold, test_data):
    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'test'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)

    fname_lst, images_lst = test_data
    pred_lst = []
    for fname, image in zip(fname_lst, images_lst):
        pred = predictor.predict(image)

        pred_path = fold_prediction_dir / f'{fname}.npy'
        np.save(pred_path, pred)

        pred = pred.mean(axis=0)
        pred_lst.append(pred)

    preds = np.stack(pred_lst, axis=0)
    subm_df = pd.DataFrame(data=preds,
                           index=fname_lst,
                           columns=config.classes)
    subm_df.index.name = 'fname'
    subm_df.to_csv(fold_prediction_dir / 'probs.csv')


def blend_test_predictions():
    probs_df_lst = []
    for fold in config.folds:
        fold_probs_path = PREDICTION_DIR / f'fold_{fold}' / 'test' / 'probs.csv'
        probs_df = pd.read_csv(fold_probs_path)
        probs_df.set_index('fname', inplace=True)
        probs_df_lst.append(probs_df)

    blend_df = gmean_preds_blend(probs_df_lst)

    if config.kernel:
        blend_df.to_csv('submission.csv')
    else:
        blend_df.to_csv(PREDICTION_DIR / 'probs.csv')


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
    transforms = get_transforms(False, CROP_SIZE)
    test_data = get_test_data()

    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = EXPERIMENT_DIR / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = Predictor(model_path, transforms,
                              BATCH_SIZE,
                              (config.audio.n_mels, CROP_SIZE),
                              (config.audio.n_mels, CROP_SIZE//4),
                              device=DEVICE)

        if not config.kernel:
            print("Val predict")
            pred_val_fold(predictor, fold)

        print("Test predict")
        pred_test_fold(predictor, fold, test_data)

    print("Blend folds predictions")
    blend_test_predictions()

    if not config.kernel:
        print("Calculate lwlrap metric on cv")
        calc_lwlrap_on_val()
