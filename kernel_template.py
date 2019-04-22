import gzip
import base64
import os
import sys
import logging
from pathlib import Path
from typing import Dict


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && export MODE=kernel && ' + command)


run('cd argus && python setup.py install')
run('python make_folds.py')
run('python train_folds.py')
