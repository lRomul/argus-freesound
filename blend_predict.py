import re
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from pathlib import Path
import multiprocessing as mp

from src.predictor import Predictor
from src.audio import read_as_melspectrogram
from src.transforms import get_transforms
from src import config


EXPERIMENTS = [
    'lsoft_008',
    'lsoft_012',
    'augment_audio_002'
]

DEVICE = 'cuda'
CROP_SIZE = 256
BATCH_SIZE = 16
N_WORKERS = mp.cpu_count()


def gmean_preds_blend(probs_df_lst):
    blend_values = np.stack([df.values for df in probs_df_lst], axis=0)
    blend_values = gmean(blend_values, axis=0)

    blend_df = probs_df_lst[0]
    blend_df.values[:] = blend_values

    return blend_df


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


def get_test_data():
    fname_lst = []
    wav_path_lst = []
    for wav_path in config.test_dir.glob('*.wav'):
        wav_path_lst.append(wav_path)
        fname_lst.append(wav_path.name)

    with mp.Pool(N_WORKERS) as pool:
        images_lst = pool.map(read_as_melspectrogram, wav_path_lst)

    return fname_lst, images_lst


def pred_test(predictor, test_data):
    fname_lst = []
    pred_lst = []
    for fname, image in zip(*test_data):
        pred = predictor.predict(image)
        pred_lst.append(pred)
        fname_lst.append(fname)

    preds = np.stack(pred_lst, axis=0)
    pred_df = pd.DataFrame(data=preds,
                           index=fname_lst,
                           columns=config.classes)
    pred_df.index.name = 'fname'

    return pred_df


def experiment_pred(experiment_dir, test_data):
    print(f"Start predict: {experiment_dir}")
    transforms = get_transforms(False, CROP_SIZE)

    pred_df_lst = []
    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = experiment_dir / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = Predictor(model_path, transforms,
                              BATCH_SIZE,
                              (config.audio.n_mels, CROP_SIZE),
                              (config.audio.n_mels, CROP_SIZE//4),
                              device=DEVICE)

        pred_df = pred_test(predictor, test_data)
        pred_df_lst.append(pred_df)

    pred_df = gmean_preds_blend(pred_df_lst)
    return pred_df


if __name__ == "__main__":
    print("Start load test data")
    test_data = get_test_data()

    exp_pred_df_lst = []
    for experiment in EXPERIMENTS:
        experiment_dir = config.experiments_dir / experiment
        exp_pred_df = experiment_pred(experiment_dir, test_data)
        exp_pred_df_lst.append(exp_pred_df)

    blend_pred_df = gmean_preds_blend(exp_pred_df_lst)
    blend_pred_df.to_csv('submission.csv')
