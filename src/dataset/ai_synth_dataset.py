import glob
import os
import os.path
import pandas as pd
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

    def __init__(self, data_dir: str):

        params_pickle_path = os.path.join(data_dir, 'params_dataset.pkl')
        self.audio_dir = os.path.join(data_dir, 'wav_files')

        self.params = pd.read_pickle(params_pickle_path)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        params_dic = self._get_audio_params(index)
        audio_path = self._get_audio_path(index)

        signal, _ = torchaudio.load(audio_path)
        signal = signal.squeeze()

        return signal, params_dic, index

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
    1. The raw audio is returned represented as PCM
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
        signal = signal.squeeze()

        return signal, index

    def _get_audio_path(self, index):
        path = self.wav_files[index]
        return path
