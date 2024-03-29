import glob
import os
import os.path
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset


class AiSynthDataset(Dataset):
    """
    A custom AI-synth dataset.
    Holds a path for the sound files, and the corresponding parameters used to create each sound

    Upon using dataloader:
    1. The raw audio is returned represented as PCM
    2. The non-numeric parameters are translated to integers
    3. All data is saved as GPU tensors
    """

    def __init__(self, data_dir: str, noise_std: float = 0.0):

        params_pickle_path = os.path.join(data_dir, 'params_dataset.pkl')
        self.audio_dir = os.path.join(data_dir, 'wav_files')

        self.params = pd.read_pickle(params_pickle_path)

        self.noise_std = noise_std

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        params_dic = self._get_audio_params(index)
        audio_path = self._get_audio_path(index)

        signal, _ = torchaudio.load(audio_path)
        signal = signal.squeeze()

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, signal.shape).astype(np.float32)
            signal = signal + noise

        # convert params to float32, since pandas defaults to float64
        params_dic = self._convert_to_float32(params_dic)

        return signal, params_dic, index

    def _convert_to_float32(self, item):
        if isinstance(item, dict):
            return {k: self._convert_to_float32(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._convert_to_float32(v) for v in item]
        elif isinstance(item, (float, int)):
            return np.float32(item)
        else:
            return item  # if it's not a type we handle, return the item as-is

    # And use it like this:

    def _get_audio_path(self, index):
        audio_file_name = f"sound_{index}.wav"

        cwd = os.getcwd()
        path = os.path.join(cwd, self.audio_dir, audio_file_name)
        return path

    def _get_audio_params(self, index):
        """
        Return audio parameters from csv.

        :param index: the index of the audio file
        :return: parameters dictionary, containing values for each parameter
        """
        params_pd_series = self.params.iloc[index]
        cells_dict = params_pd_series.to_dict()

        return cells_dict


class NSynthDataset(Dataset):
    """
    A custom NSynth dataset.
    Holds a path for the sound files

    Upon using dataloader:
    1. The raw audio is transformed from representation as PCM to [-1,1] range
    2. The non-numeric parameters are translated to integers
    3. All data is saved as GPU tensors
    """

    def __init__(self, data_dir: str):

        self.wav_files = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
        print(f"NSynth dataloader found {len(self.wav_files)} wav files in {data_dir}")

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)

        signal, _ = torchaudio.load(audio_path)
        signal = signal / 32768.0  # transform NSynth to range [-1,1]
        signal = signal.squeeze()

        return signal, index

    def _get_audio_path(self, index):
        path = self.wav_files[index]
        return path
