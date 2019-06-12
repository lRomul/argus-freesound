# Argus solution Freesound Audio Tagging 2019

Source code of 6th place solution [public LB] for [Freesound Audio Tagging 2019](https://www.kaggle.com/c/freesound-audio-tagging-2019).

## Quick setup and start 

### Requirements 

*  Nvidia drivers, CUDA >= 10.0, cuDNN >= 7
*  [Docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

The provided dockerfile is supplied to build image with CUDA support and cuDNN.


### Preparations 

* Clone the repo, build docker image. 
    ```bash
    git clone https://github.com/lRomul/argus-freesound.git
    cd argus-freesound
    make build
    ```

* Download and extract [dataset](https://www.kaggle.com/c/freesound-audio-tagging-2019/data) to `data` folder

    Folder structure should be:
    ```
    data
    ├── README.md
    ├── sample_submission.csv
    ├── test
    ├── train_curated
    ├── train_curated.csv
    ├── train_noisy
    └── train_noisy.csv
    ```

### Run

* Run docker container 
    ```bash
    make run
    ```

* Create file with folds split
    ```bash
    python make_folds.py
    ```
 
#### Single model

For example take experiment `corr_noisy_007`:
 
* Train single 5 fold model
    
    ```bash
    python train_folds.py --experiment corr_noisy_007
    ```
    
    Model weights will be in `data/experiments/corr_noisy_007`
    
* Predict train and test, evaluate metrics 

    ```bash
    python predict_folds.py --experiment corr_noisy_007
    ```
   
   Predictions, submission file and validation metrics will be in `data/predictions/corr_noisy_007`
