import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.transforms import get_transforms
from src.utils import get_best_model_path, gmean_preds_blend
from src.datasets import get_test_data
from src import config


EXPERIMENTS = [
    'auxiliary_009',
    'auxiliary_010',
    'auxiliary_011'
]

DEVICE = 'cuda'
CROP_SIZE = 256
BATCH_SIZE = 16


def pred_test(predictor, test_data):
    fname_lst, images_lst = test_data
    pred_lst = []
    for image in images_lst:
        pred = predictor.predict(image)

        pred = pred.mean(axis=0)
        pred_lst.append(pred)

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
    print("Experiments", EXPERIMENTS)
    test_data = get_test_data()

    exp_pred_df_lst = []
    for experiment in EXPERIMENTS:
        experiment_dir = config.experiments_dir / experiment
        exp_pred_df = experiment_pred(experiment_dir, test_data)
        exp_pred_df_lst.append(exp_pred_df)

    blend_pred_df = gmean_preds_blend(exp_pred_df_lst)
    blend_pred_df.to_csv('submission.csv')
