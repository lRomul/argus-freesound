import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from pathlib import Path

from src.predictor import Predictor
from src.audio import read_as_melspectrogram
from src.transforms import get_transforms
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
    pass


def pred_test_fold(predictor, fold):
    fname_lst = []
    pred_lst = []
    for wav_path in config.test_dir.glob('*.wav'):
        image = read_as_melspectrogram(wav_path)
        pred = predictor.predict(image)
        pred_lst.append(pred)
        fname_lst.append(wav_path.name)

    preds = np.stack(pred_lst, axis=0)
    subm_df = pd.DataFrame(data=preds,
                           index=fname_lst,
                           columns=config.classes)
    subm_df.index.name = 'fname'

    fold_prediction_dir = PREDICTION_DIR / f'fold_{fold}' / 'test'
    fold_prediction_dir.mkdir(parents=True, exist_ok=True)
    subm_df.to_csv(fold_prediction_dir / 'probs.csv')


def get_best_model_path(dir_path: Path):
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = score.group(0)[1:-4]
            model_scores.append((model_path, score))
    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_path = model_score[-1][0]
    return best_model_path


def blend_test_predictions():
    probs_df_lst = []
    for fold in config.folds:
        fold_probs_path = PREDICTION_DIR / f'fold_{fold}' / 'test' / 'probs.csv'
        probs_df = pd.read_csv(fold_probs_path)
        probs_df.set_index('fname', inplace=True)
        probs_df_lst.append(probs_df)

    blend_values = np.stack([df.values for df in probs_df_lst], axis=0)
    blend_values = gmean(blend_values, axis=0)

    blend_df = probs_df_lst[0]
    blend_df.values[:] = blend_values

    if config.kernel:
        blend_df.to_csv('submission.csv')
    else:
        blend_df.to_csv(PREDICTION_DIR / 'probs.csv')


if __name__ == "__main__":
    transforms = get_transforms(False, CROP_SIZE)

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
        pred_test_fold(predictor, fold)

    print("Blend folds predictions")
    blend_test_predictions()
