import gdown
import numpy as np
import sounddevice

from src.predictor import Predictor
from src.transforms import get_transforms
from src.audio import audio_to_melspectrogram
from src import config

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

config.kernel = True


model_path = config.experiments_dir / 'corr_noisy_007/fold_0/model-072-0.886906.pth'
if not model_path.exists():
    model_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download('https://drive.google.com/uc?id=1vf6LtRHlpxCC-CjyCguM4JCrso2v3Tip',
                   str(model_path), quiet=False)

DEVICE = 'cpu'
CROP_SIZE = 256
BATCH_SIZE = 16
TILE_STEP = 2

PREDICTOR = Predictor(model_path,
                      get_transforms(False, CROP_SIZE),
                      BATCH_SIZE,
                      (config.audio.n_mels, CROP_SIZE),
                      (config.audio.n_mels, CROP_SIZE // TILE_STEP),
                      device=DEVICE)

signal_block_size = config.audio.sampling_rate
SPEC_BLOCK_SIZE = 64

spec_num = 4
SPEC_LST = [np.zeros((config.audio.n_mels, SPEC_BLOCK_SIZE),
                     dtype=np.float32)] * spec_num
PREV_SIGNAL = np.zeros(signal_block_size, dtype=np.float32)


def audio_callback(indata, frames, time, status):
    global PREV_SIGNAL
    global SPEC_LST
    indata = indata.ravel()

    signal = np.concatenate([PREV_SIGNAL, indata], axis=0)
    PREV_SIGNAL = indata
    spec = audio_to_melspectrogram(signal)
    spec = spec[:, -SPEC_BLOCK_SIZE:]
    SPEC_LST = SPEC_LST[1:] + [spec]


def update_spec_plot(frame):
    spec = np.concatenate(SPEC_LST, axis=1)
    pred = PREDICTOR.predict(spec)[0]

    top3 = np.argsort(pred)[::-1][:3]
    top3 = [f"{config.classes[idx].rjust(20)} - {pred[idx]:.2f}" for idx in top3]
    print("\t".join(top3))
    im = plt.imshow(spec)
    return [im]


if __name__ == "__main__":
    fig = plt.figure(figsize=(14, 7))
    plt.title("Mel-spectrogram")

    stream = sounddevice.InputStream(callback=audio_callback,
                                     channels=1,
                                     blocksize=signal_block_size,
                                     samplerate=config.audio.sampling_rate)
    animation = FuncAnimation(fig, update_spec_plot, interval=1000, blit=True)
    with stream:
        plt.show()
