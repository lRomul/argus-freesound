#!/usr/bin/env python3
import os
import base64
import gzip
from pathlib import Path


IGNORE_LIST = ["data", "build"]
PACKAGES = ['argus']


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def check_ignore(path: Path):
    for ignore in IGNORE_LIST + PACKAGES:
        if str(path).startswith(ignore):
            return False
    return True


def build_script():
    to_encode = [p for p in Path('.').glob('**/*.py') if check_ignore(p)]
    for package in PACKAGES:
        to_encode += [p for p in Path(package).glob('**/*') if p.is_file()]

    file_data = {str(path): encode_file(path) for path in to_encode}
    print("Encoded python files:")
    for path in file_data:
        print(path)
    template = Path('kernel_template.py').read_text('utf8')
    Path('build/script.py').write_text(
        template.replace('{file_data}', str(file_data)),
        encoding='utf8')


if __name__ == '__main__':
    os.system('rm -rf argus')
    os.system('git clone https://github.com/lRomul/argus.git')
    os.system('rm -rf argus/.git')

    os.system('rm -rf build && mkdir build')
    build_script()
