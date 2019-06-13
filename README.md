# Argus solution Freesound Audio Tagging 2019

![spectrograms](readme_images/spectrograms.png)

Source code of 6th place solution [public LB] for [Freesound Audio Tagging 2019](https://www.kaggle.com/c/freesound-audio-tagging-2019). Target of competition is develop an algorithm to tag audio data automatically using a diverse vocabulary of 80 categories.

## Solution 

Key points:
* Log-scaled mel-spectrograms
* CNN model with attention and skip connections
* SpecAugment, Mixup augmentations 
* Hand relabeling curated dataset samples with low score
* Ensemble with MLP second-level model and geometric mean blending

### Data preprocessing

[Converting audio to log mel spectrograms](src/audio.py) inspired from [daisukelab's notebooks](https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data). Audio config parameters: 
```
sampling_rate = 44100
hop_length = 345 * 2
fmin = 20
fmax = sampling_rate // 2
n_mels = 128
n_fft = n_mels * 20
min_seconds = 0.5
```

### Augmentations 
Several [augmentations](src/transforms.py) was applied on spectrograms while training: 

```
size = 256
transforms = Compose([
    OneOf([
        PadToSize(size, mode='wrap'),      # Reapeat small clips
        PadToSize(size, mode='constant'),  # Pad with a minimum value
    ], p=[0.5, 0.5]),
    RandomCrop(size),                      # Crop 256 values on time axis 
    UseWithProb(
        # Random resize crop helps a lot, but I can't explain why ¯\_(ツ)_/¯   
        RandomResizedCrop(scale=(0.8, 1.0), ratio=(1.7, 2.3)),
        prob=0.33
    ),
    # Masking blocks of frequency channels, and masking blocks of time steps
    UseWithProb(SpecAugment(num_mask=2,       
                            freq_masking=0.15,
                            time_masking=0.20), 0.5),
    # Use librosa.feature.delta with order 1 and 2 for creating 2 additional channels 
    # then divide by 100
    ImageToTensor()                        
])
```

Some augmented spectrograms, looks crazy :)  
![augmentations](readme_images/augmentations.png)

### Model 

### Training 

### Ensemble 


## Quick setup and start 

### Requirements 

*  Nvidia drivers, CUDA >= 10.0, cuDNN >= 7
*  [Docker](https://www.docker.com), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

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
