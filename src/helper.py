import torch
import torchaudio
import synth
from config import TWO_PI


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    return device


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")

# class TwoWayDict(dict):
#     def __len__(self):
#         return dict.__len__(self) / 2
#
#     def __setitem__(self, key, value):
#         dict.__setitem__(self, key, value)
#         dict.__setitem__(self, value, key)


mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=synth.SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)


def map_classification_params_to_ints(classification_params_dic: dict):
    """ map classification params to ints, to input them for a neural network """

    for key, val in classification_params_dic.items():
        if key == "osc1_freq" or key == "osc2_freq":
            classification_params_dic[key] = torch.tensor([synth.OSC_FREQ_DIC[round(x.item(), 4)] for x in val])
        if isinstance(val, list):
            if "wave" in key:
                classification_params_dic[key] = torch.tensor([synth.WAVE_TYPE_DIC[x] for x in val])
            elif "filter_type" == key:
                classification_params_dic[key] = torch.tensor([synth.FILTER_TYPE_DIC[x] for x in val])


def map_classification_params_from_ints(classification_params_dic: dict):
    """ map classification params from ints (inverse operation of map_classification_params_to_ints),
     to input them for a neural network """

    for key, val in classification_params_dic.items():
        if key == "osc1_freq" or key == "osc2_freq":
            # classification_params_dic[key] = synth.OSC_FREQ_DIC[round(val, 4)]
            classification_params_dic[key] = torch.tensor([synth.OSC_FREQ_DIC_INV[x.item()] for x in val])
        if "wave" in key:
            classification_params_dic[key] = [synth.WAVE_TYPE_DIC_INV[x.item()] for x in val]
        elif "filter_type" == key:
            classification_params_dic[key] = [synth.FILTER_TYPE_DIC_INV[x.item()] for x in val]


def clamp_regression_params(parameters_dict: dict):
    """ clamp regression parameters to values that can be interpreted by the synth module"""
    '''
    ['osc1_amp', 'osc1_mod_index', 'lfo1_freq', 'lfo1_phase',
     'osc2_amp', 'osc2_mod_index', 'lfo2_freq', 'lfo2_phase',
     'filter_freq', 'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']'''

    parameters_dict['osc1_amp'] = torch.clamp(parameters_dict['osc1_amp'], min=0, max=synth.MAX_AMP)
    parameters_dict['osc1_mod_index'] = torch.clamp(parameters_dict['osc1_mod_index'], min=0, max=synth.MAX_MOD_INDEX)
    parameters_dict['lfo1_freq'] = torch.clamp(parameters_dict['lfo1_freq'], min=0, max=synth.MAX_LFO_FREQ)
    parameters_dict['lfo1_phase'] = parameters_dict['lfo1_freq'] % TWO_PI
    parameters_dict['osc2_amp'] = torch.clamp(parameters_dict['osc2_amp'], min=0, max=synth.MAX_AMP)
    parameters_dict['osc2_mod_index'] = torch.clamp(parameters_dict['osc2_mod_index'], min=0, max=synth.MAX_MOD_INDEX)
    parameters_dict['lfo2_freq'] = torch.clamp(parameters_dict['lfo2_freq'], min=0, max=synth.MAX_LFO_FREQ)
    parameters_dict['lfo2_phase'] = parameters_dict['lfo2_freq'] % TWO_PI

    parameters_dict['filter_freq'] = torch.clamp(parameters_dict['filter_freq'],
                                                 min=synth.MIN_FILTER_FREQ,
                                                 max=synth.MAX_FILTER_FREQ)

    # todo: the sum of ADSR shall not be greater than signal duration. implement the restriction
    parameters_dict['attack_t'] = torch.clamp(parameters_dict['attack_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    parameters_dict['decay_t'] = torch.clamp(parameters_dict['decay_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    parameters_dict['sustain_t'] = torch.clamp(parameters_dict['release_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    parameters_dict['release_t'] = torch.clamp(parameters_dict['release_t'], min=0, max=synth.SIGNAL_DURATION_SEC)

    parameters_dict['sustain_level'] = torch.clamp(parameters_dict['sustain_level'], min=0, max=synth.MAX_AMP)

