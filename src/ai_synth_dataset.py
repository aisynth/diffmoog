import torch
import pandas as pd
import torchaudio
import os
import helper
import synth_modules
from torch.utils.data import Dataset
from src.config import TRAIN_PARAMETERS_FILE, TRAIN_AUDIO_DIR, TEST_PARAMETERS_FILE, TEST_AUDIO_DIR, \
    DATASET_MODE, DATASET_TYPE, OS
import time


class AiSynthDataset(Dataset):
    """
    A custom Ai-synth dataset.
    Holds a path for the sound files, and the corresponding parameters used to create each sound

    Upon using dataloader:
    1. The raw audio is returned represented as log mel-spectrogram
    2. The non-numeric parameters are translated to integers
    3. All data is saved as GPU tensors
    """

    def __init__(self, dataset_mode, dataset_type, parameters_csv, audio_dir, transformation, target_sample_rate, device_arg):
        self.params = pd.read_csv(parameters_csv)
        self.audio_dir = audio_dir
        self.device = device_arg
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.dataset_mode = dataset_mode
        self.dataset_type = dataset_type

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
        classification_params_dic = helper.map_classification_params_to_ints(classification_params_dic)

        # move all synth parameters to tensors on the GPU
        for key, value in regression_params_dic.items():
            regression_params_dic[key] = torch.tensor(regression_params_dic[key]).to(helper.get_device())
        for key, value in classification_params_dic.items():
            classification_params_dic[key] = torch.tensor(classification_params_dic[key]).to(helper.get_device())

        params_dic = {
            'classification_params': classification_params_dic,
            'regression_params': regression_params_dic
        }
        return params_dic


class AiSynthSingleOscDataset(Dataset):
    """
    A custom Ai-synth dataset.
    Holds a path for the sound files, and the corresponding parameters used to create each sound

    Upon using dataloader:
    1. The raw audio is returned represented as log mel-spectrogram
    2. The non-numeric parameters are translated to integers
    3. All data is saved as GPU tensors
    """

    def __init__(self, dataset_mode, dataset_type, parameters_csv, audio_dir, target_sample_rate, device_arg):
        self.params = pd.read_csv(parameters_csv)
        self.audio_dir = audio_dir
        self.device = device_arg
        self.target_sample_rate = target_sample_rate
        self.dataset_mode = dataset_mode
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):

        params_dic = self._get_audio_params(index)
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
        classification_params_dic = helper.map_classification_params_to_ints(classification_params_dic)

        # move all synth parameters to tensors on the GPU
        for key, value in regression_params_dic.items():
            regression_params_dic[key] = torch.tensor(regression_params_dic[key]).to(helper.get_device())
        for key, value in classification_params_dic.items():
            classification_params_dic[key] = torch.tensor(classification_params_dic[key]).to(helper.get_device())

        params_dic = {
            'classification_params': classification_params_dic,
            'regression_params': regression_params_dic
        }
        return params_dic


if __name__ == "__main__":
    device = helper.get_device()

    # init dataset
    ai_synth_dataset = AiSynthDataset(DATASET_MODE,
                                      DATASET_TYPE,
                                      TRAIN_PARAMETERS_FILE,
                                      TRAIN_AUDIO_DIR,
                                      helper.mel_spectrogram_transform,
                                      synth.SAMPLE_RATE,
                                      device
                                      )

    print(f"there are {len(ai_synth_dataset)} files in the dataset")
