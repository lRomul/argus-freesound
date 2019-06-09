import gzip
import base64
import os
from pathlib import Path
from typing import Dict

EXPERIMENT_NAME = 'corr_noisy_007'
KERNEL_MODE = "predict"  # "train" or "predict"

# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && '
              f'export MODE={KERNEL_MODE} && ' + command)


run('python make_folds.py')
if KERNEL_MODE == "train":
    run(f'python train_folds.py --experiment {EXPERIMENT_NAME}')
else:
    run(f'python predict_folds.py --experiment {EXPERIMENT_NAME}')
run('rm -rf argus src')
