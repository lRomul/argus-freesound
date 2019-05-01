import numpy as np

import librosa
import librosa.display

from src.config import audio as config


def get_audio_config():
    return config.get_config_dict()


def read_audio(file_path):
    y, sr = librosa.load(file_path, sr=config.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
    min_samples = config.min_seconds * config.sampling_rate
    if len(y) < min_samples:
        padding = min_samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, padding - offset), 'constant')
    return y


def audio_to_melspectrogram(audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=config.sampling_rate,
                                                 n_mels=config.n_mels,
                                                 hop_length=config.hop_length,
                                                 n_fft=config.n_fft,
                                                 fmin=config.fmin,
                                                 fmax=config.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def show_melspectrogram(mels, title='Log-frequency power spectrogram'):
    import matplotlib.pyplot as plt

    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=config.sampling_rate, hop_length=config.hop_length,
                             fmin=config.fmin, fmax=config.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def read_as_melspectrogram(file_path, debug_display=False):
    x = read_audio(file_path)
    mels = audio_to_melspectrogram(x)
    if debug_display:
        import IPython
        IPython.display.display(IPython.display.Audio(x, rate=config.sampling_rate))
        show_melspectrogram(mels)
    return mels


if __name__ == "__main__":
    x = read_as_melspectrogram(config.train_curated_dir / '0b9906f7.wav')
    print(x.shape)
