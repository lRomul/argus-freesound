#!/usr/bin/env python3
import os
import base64
import gzip
from pathlib import Path


IGNORE_LIST = ["data", "build"]

PACKAGES = [
    'https://github.com/lRomul/argus.git'
]


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def check_ignore(path: Path, ignore_list):
    if not path.is_file():
        return False
    for ignore in ignore_list:
        if str(path).startswith(ignore):
            return False
    return True


def clone_package(git_url):
    name = Path(git_url).stem
    os.system(f'rm -rf {name}')
    os.system(f'git clone {git_url}')
    os.system(f'rm -rf {name}/.git')


def build_script(ignore_list, packages):
    to_encode = []

    for path in Path('.').glob('**/*.py'):
        if check_ignore(path, ignore_list + packages):
            to_encode.append(path)

    for package in packages:
        clone_package(package)
        package_name = Path(package).stem
        for path in Path(package_name).glob('**/*'):
            if check_ignore(path, ignore_list):
                to_encode.append(path)

    file_data = {str(path): encode_file(path) for path in to_encode}
    print("Encoded python files:")
    for path in file_data:
        print(path)
    template = Path('kernel_template.py').read_text('utf8')
    Path('kernel/script.py').write_text(
        template.replace('{file_data}', str(file_data)),
        encoding='utf8')


if __name__ == '__main__':
    os.system('rm -rf kernel && mkdir kernel')
    build_script(IGNORE_LIST, PACKAGES)
