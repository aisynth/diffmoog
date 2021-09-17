import torch
import pandas as pd
import torchaudio
import os
import helper
import synth
from torch.utils.data import Dataset
from config import PARAMETERS_FILE, DATASET_MODE
import time


class AiSynthDataset(Dataset):

    def __init__(self, csv_file, device_arg, dataset_mode):

        self.device = device_arg
        self.dataset_mode = dataset_mode
        self.params = pd.read_csv(csv_file)

        if dataset_mode == 'WAV':
            self.audio_dir = "dataset/wav_files"
            self.transformation = helper.log_mel_spec_transform.to(self.device)

        elif dataset_mode == 'MEL_SPEC':
            self.audio_dir = "dataset/audio_mel_spec_files"

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):

        params_dic = self._get_audio_params(index)
        audio_path = self._get_audio_path(index)

        if self.dataset_mode == 'WAV':
            signal, _ = torchaudio.load(audio_path)
            signal = signal.to(self.device)
            transformed_signal = self.transformation(signal)

        elif self.dataset_mode == 'MEL_SPEC':
            transformed_signal = torch.load(audio_path)

        return transformed_signal, params_dic

    def _get_audio_path(self, index):
        if self.dataset_mode == 'WAV':
            audio_file_name = f"sound_{index}.wav"
        elif self.dataset_mode == 'MEL_SPEC':
            audio_file_name = f"sound_{index}.pt"

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
        params_pd_series = params_pd_series.drop(labels=["Unnamed: 0"])

        regression_params_pd_series = params_pd_series.loc[synth.REGRESSION_PARAM_LIST]
        regression_params_dic = regression_params_pd_series.to_dict()

        classification_params_pd_series = params_pd_series.loc[synth.CLASSIFICATION_PARAM_LIST]
        classification_params_dic = classification_params_pd_series.to_dict()

        params_dic = {
            'classification_params': classification_params_dic,
            'regression_params': regression_params_dic
        }
        return params_dic


if __name__ == "__main__":
    device = helper.get_device()

    # init dataset
    ai_synth_dataset = AiSynthDataset(csv_file=PARAMETERS_FILE,
                                      device_arg=device,
                                      dataset_mode=DATASET_MODE,
                                      transformation=helper.mel_spectrogram_transform,
                                      )

    print(f"there are {len(ai_synth_dataset)} files in the dataset")
