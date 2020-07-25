#!/usr/bin/env bash
set -e
NAME="argus-freesound"
DOCKER_OPTIONS="--rm -it --gpus=all --ipc=host -v $(pwd):/workdir --name=${NAME} ${NAME}"

git checkout 7ce37a28c80412c8acc18a1773badc7879d1c054
docker build -t ${NAME} .

# Build kernel
git checkout ddbe02ae88b6bd05c1b9726d2fd30c38854be4fd
docker run ${DOCKER_OPTIONS} python build_kernel.py

# Make folds split
docker run ${DOCKER_OPTIONS} python make_folds.py

# Experiment auxiliary_016
git checkout 31156c79e470ffacc494ba846aef3bd80faf0d10
docker run ${DOCKER_OPTIONS} python train_folds.py --experiment auxiliary_016

# Experiment auxiliary_019
git checkout 9639288b9240e7e45db497feb7593f05a4f463d1
docker run ${DOCKER_OPTIONS} python train_folds.py --experiment auxiliary_019

# Experiment corr_noisy_003
git checkout 1fb2eea443d99df4538420fa42daf098c94322c2
docker run ${DOCKER_OPTIONS} python train_folds.py --experiment corr_noisy_003

# Experiment corr_noisy_004
git checkout db945ac11df559e0e1c0a2be464faf46122f1bef
docker run ${DOCKER_OPTIONS} python train_folds.py --experiment corr_noisy_004

# Experiment corr_noisy_007
git checkout bdb9150146ad8d500b4e19fa6b9fe98111fb28b0
docker run ${DOCKER_OPTIONS} python train_folds.py --experiment corr_noisy_007

# Experiment corrections_002
git checkout 05a7aee7c50148677735531bdddf32902b468bea
docker run ${DOCKER_OPTIONS} python train_folds.py --experiment corrections_002

# Experiment corrections_003
git checkout 24a4f20ffc284d22b38bbabfe510ed194f62e496
docker run ${DOCKER_OPTIONS} python train_folds.py --experiment corrections_003


# Experiment stacking_008_fcnet_43040
git checkout 1e1c265fc6e45c103d8d741c1bdcc5959f71348d
docker run ${DOCKER_OPTIONS} python train_stacking.py

# Stacking train stacking_008_fcnet_45041
git checkout bc48f8a17ac4452ee3f2a3d18fd7caa31f812b27
docker run ${DOCKER_OPTIONS} python train_stacking.py

# Stacking train stacking_008_fcnet_50013
git checkout 493908aeaff4b0e1df8298003b10af1cf56e6b3c
docker run ${DOCKER_OPTIONS} python train_stacking.py

git checkout master
