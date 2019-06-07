import os
import json
from pathlib import Path
from hashlib import sha1


kernel = False
kernel_mode = ""
if 'MODE' in os.environ:
    kernel = True
    kernel_mode = os.environ['MODE']
    assert kernel_mode in ["train", "predict"]

if kernel:
    if kernel_mode == "train":
        input_data_dir = Path('/kaggle/input/')
    else:
        input_data_dir = Path('/kaggle/input/freesound-audio-tagging-2019/')
    save_data_dir = Path('/kaggle/working/')
else:
    input_data_dir = Path('/workdir/data/')
    save_data_dir = Path('/workdir/data/')

train_curated_dir = input_data_dir / 'train_curated'
train_noisy_dir = input_data_dir / 'train_noisy'
train_curated_csv_path = input_data_dir / 'train_curated.csv'
train_noisy_csv_path = input_data_dir / 'train_noisy.csv'
test_dir = input_data_dir / 'test'
sample_submission = input_data_dir / 'sample_submission.csv'

train_folds_path = save_data_dir / 'train_folds.csv'
predictions_dir = save_data_dir / 'predictions'
if kernel and kernel_mode == "predict":
    def find_kernel_data_dir():
        kaggle_input = Path('/kaggle/input/')
        train_kernel_name = 'freesound-train'
        default = kaggle_input / train_kernel_name
        if default.exists():
            return default
        else:
            for path in kaggle_input.glob('*'):
                if path.is_dir():
                    if path.name.startswith(train_kernel_name):
                        return path
        return default
    experiments_dir = find_kernel_data_dir() / 'experiments'
else:
    experiments_dir = save_data_dir / 'experiments'

folds_data_pkl_dir = save_data_dir / 'folds_data'
augment_folds_data_pkl_dir = save_data_dir / 'augment_folds_data'
noisy_data_pkl_dir = save_data_dir / 'noisy_data'
corrections_json_path = Path('/workdir/corrections.json')
noisy_corrections_json_path = Path('/workdir/noisy_corrections.json')

n_folds = 5
folds = list(range(n_folds))


class audio:
    sampling_rate = 44100
    hop_length = 345 * 2
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    min_seconds = 0.5

    @classmethod
    def get_config_dict(cls):
        config_dict = dict()
        for key, value in cls.__dict__.items():
            if key[:1] != '_' and \
                    key not in ['get_config_dict', 'get_hash']:
                config_dict[key] = value
        return config_dict

    @classmethod
    def get_hash(cls, **kwargs):
        config_dict = cls.get_config_dict()
        config_dict = {**config_dict, **kwargs}
        hash_str = json.dumps(config_dict,
                              sort_keys=True,
                              ensure_ascii=False,
                              separators=None)
        hash_str = hash_str.encode('utf-8')
        return sha1(hash_str).hexdigest()[:7]


classes = [
    'Accelerating_and_revving_and_vroom',
    'Accordion',
    'Acoustic_guitar',
    'Applause',
    'Bark',
    'Bass_drum',
    'Bass_guitar',
    'Bathtub_(filling_or_washing)',
    'Bicycle_bell',
    'Burping_and_eructation',
    'Bus',
    'Buzz',
    'Car_passing_by',
    'Cheering',
    'Chewing_and_mastication',
    'Child_speech_and_kid_speaking',
    'Chink_and_clink',
    'Chirp_and_tweet',
    'Church_bell',
    'Clapping',
    'Computer_keyboard',
    'Crackle',
    'Cricket',
    'Crowd',
    'Cupboard_open_or_close',
    'Cutlery_and_silverware',
    'Dishes_and_pots_and_pans',
    'Drawer_open_or_close',
    'Drip',
    'Electric_guitar',
    'Fart',
    'Female_singing',
    'Female_speech_and_woman_speaking',
    'Fill_(with_liquid)',
    'Finger_snapping',
    'Frying_(food)',
    'Gasp',
    'Glockenspiel',
    'Gong',
    'Gurgling',
    'Harmonica',
    'Hi-hat',
    'Hiss',
    'Keys_jangling',
    'Knock',
    'Male_singing',
    'Male_speech_and_man_speaking',
    'Marimba_and_xylophone',
    'Mechanical_fan',
    'Meow',
    'Microwave_oven',
    'Motorcycle',
    'Printer',
    'Purr',
    'Race_car_and_auto_racing',
    'Raindrop',
    'Run',
    'Scissors',
    'Screaming',
    'Shatter',
    'Sigh',
    'Sink_(filling_or_washing)',
    'Skateboard',
    'Slam',
    'Sneeze',
    'Squeak',
    'Stream',
    'Strum',
    'Tap',
    'Tick-tock',
    'Toilet_flush',
    'Traffic_noise_and_roadway_noise',
    'Trickle_and_dribble',
    'Walk_and_footsteps',
    'Water_tap_and_faucet',
    'Waves_and_surf',
    'Whispering',
    'Writing',
    'Yell',
    'Zipper_(clothing)'
]

class2index = {cls: idx for idx, cls in enumerate(classes)}
