from torch.utils.data import Dataset
from config import SAMPLE_RATE, PARAMETERS_FILE, AUDIO_DIR
import pandas as pd
import torchaudio
import os


class AiSynthDataset(Dataset):

    def __init__(self, parameters_csv, audio_dir):
        self.parameters = pd.read_csv(parameters_csv)
        print(self.parameters)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        parameters = self._get_audio_parameters(index)
        signal, sr = torchaudio.load(audio_path)
        return signal, parameters

    def _get_audio_path(self,index):
        audio_file_name = f"sound_{index}.wav"
        cwd = os.getcwd()
        path = os.path.join(cwd, self.audio_dir, audio_file_name)
        return path

    def _get_audio_parameters(self, index):
        return self.parameters.iloc[index]


if __name__ == "__main__":

    ai_synth_dataset = AiSynthDataset(PARAMETERS_FILE, AUDIO_DIR)
    print(f"there are {len(ai_synth_dataset)} files in the dataset")
    signal, parameters = ai_synth_dataset[0]
