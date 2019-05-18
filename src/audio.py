import numpy as np

import librosa
import librosa.display

from src.config import audio as config


def get_audio_config():
    return config.get_config_dict()


def read_audio(file_path):
    min_samples = int(config.min_seconds * config.sampling_rate)
    try:
        y, sr = librosa.load(file_path, sr=config.sampling_rate)
        trim_y, trim_idx = librosa.effects.trim(y)  # trim, top_db=default(60)

        if len(trim_y) < min_samples:
            center = (trim_idx[1] - trim_idx[0]) // 2
            left_idx = max(0, center - min_samples // 2)
            right_idx = min(len(y), center + min_samples // 2)
            trim_y = y[left_idx:right_idx]

            if len(trim_y) < min_samples:
                padding = min_samples - len(trim_y)
                offset = padding // 2
                trim_y = np.pad(trim_y, (offset, padding - offset), 'constant')
        return trim_y
    except BaseException as e:
        print(f"Exception while reading file {e}")
        return np.zeros(min_samples, dtype=np.float32)


def audio_to_melspectrogram(audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=config.sampling_rate,
                                                 n_mels=config.n_mels,
                                                 hop_length=config.hop_length,
                                                 n_fft=config.n_fft,
                                                 fmin=config.fmin,
                                                 fmax=config.fmax,
                                                 power=2)
    spectrogram = librosa.power_to_db(spectrogram)
    # delta = librosa.feature.delta(spectrogram, order=1)
    # accelerate = librosa.feature.delta(spectrogram, order=2)
    spectrogram -= spectrogram.min()
    spectrogram /= 80.0
    return spectrogram


def audio_to_pcen(audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=config.sampling_rate,
                                                 n_mels=config.n_mels,
                                                 hop_length=config.hop_length,
                                                 n_fft=config.n_fft,
                                                 fmin=config.fmin,
                                                 fmax=config.fmax,
                                                 power=1)
    # amplitude = librosa.amplitude_to_db(spectrogram, ref=np.max)
    pcen = librosa.pcen(spectrogram,
                        sr=config.sampling_rate,
                        hop_length=config.hop_length)
    pcen /= 4.0
    return pcen


def audio_to_tempogram(audio):
    oenv = librosa.onset.onset_strength(y=audio,
                                        sr=config.sampling_rate,
                                        hop_length=config.hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv,
                                          sr=config.sampling_rate,
                                          hop_length=config.hop_length,
                                          win_length=config.n_mels)
    return tempogram


def show_melspectrogram(mels, title='Log-frequency power spectrogram'):
    import matplotlib.pyplot as plt

    librosa.display.specshow(mels, x_axis='time', y_axis='mel',
                             sr=config.sampling_rate, hop_length=config.hop_length,
                             fmin=config.fmin, fmax=config.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def read_as_melspectrogram(file_path, time_stretch=1.0, pitch_shift=0.0,
                           debug_display=False):
    x = read_audio(file_path)
    if time_stretch != 1.0:
        x = librosa.effects.time_stretch(x, time_stretch)

    if pitch_shift != 0.0:
        librosa.effects.pitch_shift(x, config.sampling_rate, n_steps=pitch_shift)

    melspectrogram = audio_to_melspectrogram(x)
    pcen = audio_to_pcen(x)
    tempogram = audio_to_tempogram(x)

    if debug_display:
        import IPython
        IPython.display.display(IPython.display.Audio(x, rate=config.sampling_rate))
        show_melspectrogram(melspectrogram)
        show_melspectrogram(pcen)
        show_melspectrogram(tempogram)

    result = np.stack([melspectrogram, pcen, tempogram], axis=2)
    return result.astype(np.float32)


if __name__ == "__main__":
    x = read_as_melspectrogram(config.train_curated_dir / '0b9906f7.wav')
    print(x.shape)
