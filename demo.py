import numpy as np
import sounddevice

from src.predictor import Predictor
from src.transforms import get_transforms
from src.audio import audio_to_melspectrogram
from src import config

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

config.kernel = True

model_path = '/workdir/data/experiments/corr_noisy_007/fold_0/model-072-0.886906.pth'

DEVICE = 'cpu'
CROP_SIZE = 256
BATCH_SIZE = 16
TILE_STEP = 2

transforms = get_transforms(False, CROP_SIZE)

predictor = Predictor(model_path, transforms,
                      BATCH_SIZE,
                      (config.audio.n_mels, CROP_SIZE),
                      (config.audio.n_mels, CROP_SIZE // TILE_STEP),
                      device=DEVICE)

signal_block_size = config.audio.sampling_rate
spec_block_size = 64

spec_num = 4
spec_lst = [np.zeros((config.audio.n_mels, spec_block_size),
                     dtype=np.float32)] * spec_num
prev_signal = np.zeros(signal_block_size, dtype=np.float32)


def audio_callback(indata, frames, time, status):
    global prev_signal
    global spec_lst
    indata = indata.ravel()

    signal = np.concatenate([prev_signal, indata], axis=0)
    prev_signal = indata
    spec = audio_to_melspectrogram(signal)
    spec = spec[:, -spec_block_size:]
    spec_lst = spec_lst[1:] + [spec]


def update_spec_plot(frame):
    spec = np.concatenate(spec_lst, axis=1)
    pred = predictor.predict(spec)[0]

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
