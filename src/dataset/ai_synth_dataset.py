import os
import os.path
import pandas as pd
import torchaudio
from model import helper
from torch.utils.data import Dataset
from config import Config
from torch.utils.data import DataLoader


class AiSynthDataset(Dataset):
    """
    A custom AI-synth dataset.
    Holds a path for the sound files, and the corresponding parameters used to create each sound

    Upon using dataloader:
    1. The raw audio is returned represented as PCM
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
        return len(self.params)

    def __getitem__(self, index):
        params_dic = self._get_audio_params(index)
        audio_path = self._get_audio_path(index)

        signal, _ = torchaudio.load(audio_path)
        # signal = signal.to(self.device)

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

    def __init__(self,
                 audio_dir,
                 device_arg):
        self.audio_dir = audio_dir
        self.device = device_arg

    def __len__(self):
        res = len([name for name in os.listdir(self.audio_dir) if os.path.isfile(os.path.join(self.audio_dir, name))])
        return res

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        signal, _ = torchaudio.load(audio_path)
        # signal = signal.to(self.device)

        return signal, index

    def _get_audio_path(self, index):
        audio_file_name = f"sound_{index}.wav"

        cwd = os.getcwd()
        path = os.path.join(cwd, self.audio_dir, audio_file_name)
        return path


def create_data_loader(dataset, batch_size, num_workers=0, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            persistent_workers=num_workers != 0, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    device = helper.get_device()
    cfg = Config()
    # init dataset
    ai_synth_dataset = AiSynthDataset(cfg.train_parameters_file,
                                      cfg.train_audio_dir,
                                      device
                                      )

    print(f"there are {len(ai_synth_dataset)} files in the dataset")
