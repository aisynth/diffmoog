import torch.cuda
from torch.utils.data import Dataset
from config import SAMPLE_RATE, PARAMETERS_FILE, AUDIO_DIR
import pandas as pd
import torchaudio
import os


class AiSynthDataset(Dataset):

    def __init__(self, parameters_csv, audio_dir, transformation, target_sample_rate, device):
        self.parameters = pd.read_csv(parameters_csv)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        sound_parameters = self._get_audio_parameters(index)
        sound_signal, sr = torchaudio.load(audio_path)
        sound_signal = sound_signal.to(self.device)
        transformed_sound_signal = self.transformation(sound_signal)
        return transformed_sound_signal, sound_parameters

    def _get_audio_path(self, index):
        audio_file_name = f"sound_{index}.wav"
        cwd = os.getcwd()
        path = os.path.join(cwd, self.audio_dir, audio_file_name)
        return path

    def _get_audio_parameters(self, index):
        return self.parameters.iloc[index]


if __name__ == "__main__":

    if torch.cuda.is_available()():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    ai_synth_dataset = AiSynthDataset(PARAMETERS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, device)
    print(f"there are {len(ai_synth_dataset)} files in the dataset")
    signal, parameters = ai_synth_dataset[0]
