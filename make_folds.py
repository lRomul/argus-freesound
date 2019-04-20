import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from src import config


if __name__ == '__main__':
    random_state = 42

    random.seed(random_state)
    np.random.seed(random_state)

    train_curated_df = pd.read_csv(config.train_curated_csv_path)
    train_curated_df['fold'] = -1
    file_paths = train_curated_df.fname.apply(lambda x: config.train_curated_dir / x)
    train_curated_df['file_path'] = file_paths

    kf = KFold(n_splits=config.n_folds, random_state=random_state, shuffle=True)

    for fold, (_, val_index) in enumerate(kf.split(train_curated_df)):
        train_curated_df.iloc[val_index, 2] = fold

    train_curated_df.to_csv(config.train_folds_path, index=False)
    print(f"Train folds saved to '{config.train_folds_path}'")
