import re
import tqdm
import pandas as pd
from pathlib import Path

from argus import load_model

from src.audio import read_as_melspectrogram
from src.transforms import ImageToTensor
from src import config


EXPERIMENT_NAME = 'test_007'
EXPERIMENT_DIR = config.experiments_dir / EXPERIMENT_NAME
PREDICTION_DIR = config.predictions_dir / EXPERIMENT_NAME
DEVICE = 'cuda'
MAX_SIZE = 2048


class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.model = load_model(model_path, device=device)

    def __call__(self, audio):
        pass


def pred_val_fold(model, fold):
    pass


def pred_test_fold(model, fold):
    image2tensor = ImageToTensor()

    subm_df = pd.read_csv(config.sample_submission)
    subm_df.set_index('fname', inplace=True)
    subm_df = subm_df.astype(float)
    assert all(subm_df.columns == config.classes)

    for fname in tqdm.tqdm(subm_df.index):
        image = read_as_melspectrogram(config.test_dir / fname)

        if image.shape[1] > MAX_SIZE:
            print("Maximum size exceeded:", image.shape, fname)
            image = image[:, :MAX_SIZE]

        tensor = image2tensor(image)
        tensor = tensor.unsqueeze(dim=0)

        pred = model.predict(tensor)
        pred = pred.cpu().numpy()

        subm_df.loc[fname] = pred

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

    blend_df = probs_df_lst[0]
    for probs_df in probs_df_lst[1:]:
        blend_df += probs_df
    blend_df = blend_df / len(probs_df_lst)

    if config.kernel:
        blend_df.to_csv('submission.csv')
    else:
        blend_df.to_csv(PREDICTION_DIR / 'probs.csv')


if __name__ == "__main__":
    for fold in config.folds:
        print("Predict fold", fold)
        fold_dir = EXPERIMENT_DIR / f'fold_{fold}'
        model_path = get_best_model_path(fold_dir)
        print("Model path", model_path)
        model = load_model(model_path, device=DEVICE)

        if not config.kernel:
            print("Val predict")
            pred_val_fold(model_path, fold)

        print("Test predict")
        pred_test_fold(model, fold)

    print("Blend folds predictions")
    blend_test_predictions()
