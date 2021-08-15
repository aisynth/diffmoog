from torch.utils.data import Dataset
from config import PARAMETERS_FILE, AUDIO_DIR
import pandas as pd
import torchaudio
import os
import helper
import synth


class AiSynthDataset(Dataset):

    def __init__(self, parameters_csv, audio_dir, transformation, target_sample_rate, device_arg):
        self.params = pd.read_csv(parameters_csv)
        self.audio_dir = audio_dir
        self.device = device_arg
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        params_dic = self._get_audio_params(index)
        signal, sr = torchaudio.load(audio_path)
        signal = signal.to(self.device)
        transformed_signal = self.transformation(signal)
        return transformed_signal, params_dic

    def _get_audio_path(self, index):
        audio_file_name = f"sound_{index}.wav"
        cwd = os.getcwd()
        path = os.path.join(cwd, self.audio_dir, audio_file_name)
        return path

    def _get_audio_params(self, index):
        """
        Return audio parameters from csv. Classification parameters are translated to integers from mappings.

        :param index: the index of the audio file
        :return: parameters dictionary, containing values for each parameter
        """
        params_pd_series = self.params.iloc[index]
        params_pd_series = params_pd_series.drop(labels=["Unnamed: 0", "file_name"])

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
    ai_synth_dataset = AiSynthDataset(PARAMETERS_FILE,
                                      AUDIO_DIR,
                                      helper.mel_spectrogram_transform,
                                      synth.SAMPLE_RATE,
                                      device)

    print(f"there are {len(ai_synth_dataset)} files in the dataset")
