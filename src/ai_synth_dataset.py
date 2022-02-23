import torch
import pandas as pd
import torchaudio
import os
import helper
from torch.utils.data import Dataset
import ast
from config import Config, SynthConfig


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
                 dataset_mode,
                 parameters_pickle,
                 audio_dir,
                 target_sample_rate,
                 device_arg,
                 synth_cfg: SynthConfig):
        self.params = pd.read_pickle(parameters_pickle)
        self.audio_dir = audio_dir
        self.device = device_arg
        self.target_sample_rate = target_sample_rate
        self.dataset_mode = dataset_mode
        self.synth_cfg = synth_cfg

    def __len__(self):
        a = len(self.params)
        return len(self.params)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            a = 0
        if index == (0,0):
            a=0
        params_dic = self._get_audio_params(index, self.synth_cfg)
        audio_path = self._get_audio_path(index)

        if self.dataset_mode == 'WAV':
            signal, _ = torchaudio.load(audio_path)
            signal = signal.to(self.device)

        elif self.dataset_mode == 'MEL_SPEC':
            signal = torch.load(audio_path)

        return signal, params_dic

    def _get_audio_path(self, index):
        if self.dataset_mode == 'WAV':
            audio_file_name = f"sound_{index}.wav"
        elif self.dataset_mode == 'MEL_SPEC':
            audio_file_name = f"sound_{index}.pt"

        cwd = os.getcwd()
        path = os.path.join(cwd, self.audio_dir, audio_file_name)
        return path

    def _get_audio_params(self, index, synth_cfg: SynthConfig):
        """
        Return audio parameters from csv.

        :param index: the index of the audio file
        :return: parameters dictionary, containing values for each parameter
        """
        params_pd_series = self.params.iloc[index]
        # params_pd_series = params_pd_series.drop(labels=["Unnamed: 0"])

        # params_pd_series2 = self.params.iloc[2]
        # params_pd_series2= params_pd_series2.drop(labels=["Unnamed: 0"])

        cells_dict = params_pd_series.to_dict()

        return cells_dict


if __name__ == "__main__":
    device = helper.get_device()
    cfg = Config()
    # init dataset
    ai_synth_dataset = AiSynthDataset(cfg.dataset_mode,
                                      cfg.train_parameters_file,
                                      cfg.train_audio_dir,
                                      helper.mel_spectrogram_transform,
                                      cfg.sample_rate,
                                      device
                                      )

    print(f"there are {len(ai_synth_dataset)} files in the dataset")
