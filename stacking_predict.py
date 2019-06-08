import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

from src.predictor import Predictor
from src.transforms import get_transforms
from src.utils import get_best_model_path
from src.datasets import get_test_data
from src import config

from src.stacking.predictor import StackPredictor


NAME = "blend_stacks_001"

EXPERIMENTS = [
    [
        'auxiliary_001',
        'auxiliary_012',
        'auxiliary_014',
        'rnn_aux_skip_attention_001',
        'small_cat_002',
    ],
    [
        'auxiliary_001',
        'auxiliary_012',
        'corrections_002',
        'corr_noisy_001',
        'corr_noisy_002',
    ]
]
STACKING_EXPERIMENTS = [
    [
        'fcnet_stacking_007',
        'fcnet_stacking_008'
    ],
    [
        'stacking_006_fcnet_45041',
        'stacking_006_fcnet_43040'
    ]
]


DEVICE = 'cuda'
CROP_SIZE = 256
BATCH_SIZE = 16
STACK_BATCH_SIZE = 256
TILE_STEP = 2


def pred_test(predictor, images_lst):
    pred_lst = []
    for image in images_lst:
        pred = predictor.predict(image)

        pred = pred.mean(axis=0)
        pred_lst.append(pred)

    preds = np.stack(pred_lst, axis=0)
    return preds


def experiment_pred(experiment_dir, images_lst):
    print(f"Start predict: {experiment_dir}")
    transforms = get_transforms(False, CROP_SIZE)

    pred_lst = []
    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = experiment_dir / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = Predictor(model_path, transforms,
                              BATCH_SIZE,
                              (config.audio.n_mels, CROP_SIZE),
                              (config.audio.n_mels, CROP_SIZE//TILE_STEP),
                              device=DEVICE)

        pred = pred_test(predictor, images_lst)
        pred_lst.append(pred)

    preds = gmean(pred_lst, axis=0)
    return preds


def stacking_pred(experiment_dir, stack_probs):
    print(f"Start predict: {experiment_dir}")

    pred_lst = []
    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = experiment_dir / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        predictor = StackPredictor(model_path, STACK_BATCH_SIZE,
                                   device=DEVICE)
        pred = predictor.predict(stack_probs)
        pred_lst.append(pred)

    preds = gmean(pred_lst, axis=0)
    return preds


def stacks_predict(experiments, stacking_experiments):
    exp_pred_lst = []
    for experiment in experiments:
        experiment_dir = config.experiments_dir / experiment
        exp_pred = experiment_pred(experiment_dir, images_lst)
        exp_pred_lst.append(exp_pred)

    stack_probs = np.concatenate(exp_pred_lst, axis=1)

    stack_pred_lst = []
    for experiment in stacking_experiments:
        experiment_dir = config.experiments_dir / experiment
        stack_pred = stacking_pred(experiment_dir, stack_probs)
        stack_pred_lst.append(stack_pred)

    stack_pred = gmean(exp_pred_lst + stack_pred_lst, axis=0)
    return stack_pred


if __name__ == "__main__":
    print("Name", NAME)
    print("Experiments", EXPERIMENTS)
    print("Stacking experiments", STACKING_EXPERIMENTS)
    print("Device", DEVICE)
    print("Crop size", CROP_SIZE)
    print("Batch size", BATCH_SIZE)
    print("Stacking batch size", STACK_BATCH_SIZE)
    print("Tile step", TILE_STEP)

    fname_lst, images_lst = get_test_data()

    stacks_pred_lst = []
    for experiments, stacking_experiments in zip(EXPERIMENTS, STACKING_EXPERIMENTS):
        stack_pred = stacks_predict(experiments, stacking_experiments)
        stacks_pred_lst.append(stack_pred)

    stacks_pred = gmean(stacks_pred_lst, axis=0)

    stack_pred_df = pd.DataFrame(data=stacks_pred,
                                 index=fname_lst,
                                 columns=config.classes)
    stack_pred_df.index.name = 'fname'
    stack_pred_df.to_csv('submission.csv')
