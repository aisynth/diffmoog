import torch
import pandas as pd
import torchaudio
import os
import helper
from torch.utils.data import Dataset
from config import Config
from torch.utils.data import DataLoader


class AiSynthDataset(Dataset):
    """
    A custom AI-synth dataset.
    Holds a path for the sound files, and the corresponding parameters used to create each sound

    Upon using dataloader:
    1. The raw audio is returned represented as log mel-spectrogram
    2. The non-numeric parameters are translated to integers
    3. All data is saved as GPU tensors
    """

    def __init__(self,
                 parameters_pickle,
                 audio_dir,
                 device_arg):
        self.params = pd.read_pickle(parameters_pickle)
        self.audio_dir = audio_dir
        self.device = device_arg

    def __len__(self):
        a = len(self.params)
        return len(self.params)

    def __getitem__(self, index):
        params_dic = self._get_audio_params(index)
        audio_path = self._get_audio_path(index)

        signal, _ = torchaudio.load(audio_path)
        signal = signal.to(self.device)

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


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

if __name__ == "__main__":
    device = helper.get_device()
    cfg = Config()
    # init dataset
    ai_synth_dataset = AiSynthDataset(cfg.train_parameters_file,
                                      cfg.train_audio_dir,
                                      device
                                      )

    print(f"there are {len(ai_synth_dataset)} files in the dataset")
