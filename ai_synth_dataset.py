import torch.cuda
from torch.utils.data import Dataset
from config import SAMPLE_RATE, PARAMETERS_FILE, AUDIO_DIR, WAVE_TYPE_DIC, FILTER_TYPE_DIC, OSC_FREQ_DIC, \
    CLASSIFICATION_PARAMETERS_LIST, REGRESSION_PARAMETERS_LIST
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
        sound_classification_parameters, sound_regression_parameters = self._get_audio_parameters(index)
        sound_signal, sr = torchaudio.load(audio_path)
        sound_signal = sound_signal.to(self.device)
        transformed_sound_signal = self.transformation(sound_signal)
        return transformed_sound_signal, sound_classification_parameters, sound_regression_parameters

    def _get_audio_path(self, index):
        audio_file_name = f"sound_{index}.wav"
        cwd = os.getcwd()
        path = os.path.join(cwd, self.audio_dir, audio_file_name)
        return path

    def _get_audio_parameters(self, index):
        """
        Return audio parameters from csv. Classification parameters are translated to integers from mappings.

        :param index: the index of the audio file
        :return: parameters dictionary, containing values for each parameter
        """
        parameters_pd_series = self.parameters.iloc[index]
        parameters_pd_series = parameters_pd_series.drop(labels=["Unnamed: 0", "file_name"])
        regression_parameters_pd_series = parameters_pd_series.loc[REGRESSION_PARAMETERS_LIST]
        regression_parameters_pd_series = regression_parameters_pd_series.to_dict()
        classification_parameters_pd_series = parameters_pd_series.loc[CLASSIFICATION_PARAMETERS_LIST]
        classification_parameters_pd_series = classification_parameters_pd_series.to_dict()
        # map string attributes to numbers
        for key, val in classification_parameters_pd_series.items():
            if "osc1_freq" == key or "osc2_freq" == key:
                classification_parameters_pd_series[key] = OSC_FREQ_DIC[round(val, 4)]
            if isinstance(val, str):
                if "wave" in key:
                    classification_parameters_pd_series[key] = WAVE_TYPE_DIC[val]
                elif "filter_type" == key:
                    classification_parameters_pd_series[key] = FILTER_TYPE_DIC[val]
        return classification_parameters_pd_series, regression_parameters_pd_series


if __name__ == "__main__":

    if torch.cuda.is_available():
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
