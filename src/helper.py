import torch
import torchaudio
import synth


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    return device


class TwoWayDict(dict):
    def __len__(self):
        return dict.__len__(self) / 2

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)


mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=synth.SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

# todo: implement as vecor operation
def map_classification_params_to_ints(classification_params_dic):
    for key, val in classification_params_dic.items():
        if key == "osc1_freq" or key == "osc2_freq":
            classification_params_dic[key] = synth.OSC_FREQ_DIC[round(val, 4)]
        if isinstance(val, str):
            if "wave" in key:
                classification_params_dic[key] = synth.WAVE_TYPE_DIC[val]
            elif "filter_type" == key:
                classification_params_dic[key] = synth.FILTER_TYPE_DIC[val]
