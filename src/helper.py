import torch
from torch import Tensor
import torchaudio
import synth
import matplotlib.pyplot as plt
import librosa
from config import TWO_PI, DEBUG_MODE, SAMPLE_RATE, SYNTH_TYPE
from synth_config import *
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if DEBUG_MODE:
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
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
    power=2.0
).to(get_device())

amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB().to(get_device())

log_mel_spec_transform = torch.nn.Sequential(mel_spectrogram_transform, amplitude_to_db_transform).to(get_device())


def map_classification_params_to_ints(params_dic: dict):
    """ map classification params to ints, to input them for a neural network """
    mapped_params_dict = {}
    for key, val in params_dic.items():
        if key in synth.CLASSIFICATION_PARAM_LIST:
            if key == "osc1_freq" or key == "osc2_freq":
                # todo: inspect which of these are needed (i think only the middle one)
                if torch.is_tensor(val):
                    mapped_params_dict[key] = [synth.OSC_FREQ_DIC[round(x.item(), 4)] for x in val]
                if isinstance(val, float):
                    mapped_params_dict[key] = synth.OSC_FREQ_DIC[round(val, 4)]
                else:
                    mapped_params_dict[key] = [synth.OSC_FREQ_DIC[round(x, 4)] for x in val]
            if "wave" in key:
                mapped_params_dict[key] = synth.WAVE_TYPE_DIC[val]
            elif "filter_type" == key:
                mapped_params_dict[key] = synth.FILTER_TYPE_DIC[val]

    return mapped_params_dict


def map_classification_params_from_ints(params_dic: dict):
    """ map classification params from ints (inverse operation of map_classification_params_to_ints),
     to input them for a neural network """

    mapped_params_dict = {}
    for key, val in params_dic.items():
        if key in synth.CLASSIFICATION_PARAM_LIST:
            if key == "osc1_freq" or key == "osc2_freq":
                # classification_params_dic[key] = synth.OSC_FREQ_DIC[round(val, 4)]
                mapped_params_dict[key] = torch.tensor([synth.OSC_FREQ_DIC_INV[x.item()] for x in val])
            if "wave" in key:
                mapped_params_dict[key] = [synth.WAVE_TYPE_DIC_INV[x.item()] for x in val]
            elif "filter_type" == key:
                mapped_params_dict[key] = [synth.FILTER_TYPE_DIC_INV[x.item()] for x in val]

    return mapped_params_dict


def clamp_regression_params(parameters_dict: dict):
    """ clamp regression parameters to values that can be interpreted by the synth module,
        add classification parameters as is"""
    '''
    ['osc1_amp', 'osc1_mod_index', 'lfo1_freq',
     'osc2_amp', 'osc2_mod_index', 'lfo2_freq',
     'filter_freq', 'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']'''

    clamped_params_dict = {}
    clamped_params_dict['osc1_amp'] = torch.clamp(parameters_dict['osc1_amp'], min=0, max=synth.MAX_AMP)
    clamped_params_dict['osc1_mod_index'] = torch.clamp(parameters_dict['osc1_mod_index'], min=0,
                                                        max=synth.MAX_MOD_INDEX)
    clamped_params_dict['lfo1_freq'] = torch.clamp(parameters_dict['lfo1_freq'], min=0, max=synth.MAX_LFO_FREQ)
    clamped_params_dict['osc2_amp'] = torch.clamp(parameters_dict['osc2_amp'], min=0, max=synth.MAX_AMP)
    clamped_params_dict['osc2_mod_index'] = torch.clamp(parameters_dict['osc2_mod_index'], min=0,
                                                        max=synth.MAX_MOD_INDEX)
    clamped_params_dict['lfo2_freq'] = torch.clamp(parameters_dict['lfo2_freq'], min=0, max=synth.MAX_LFO_FREQ)

    clamped_params_dict['filter_freq'] = torch.clamp(parameters_dict['filter_freq'],
                                                     min=synth.MIN_FILTER_FREQ,
                                                     max=synth.MAX_FILTER_FREQ)

    attack_t = torch.clamp(parameters_dict['attack_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    decay_t = torch.clamp(parameters_dict['decay_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    sustain_t = torch.clamp(parameters_dict['release_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    release_t = torch.clamp(parameters_dict['release_t'], min=0, max=synth.SIGNAL_DURATION_SEC)

    # clamp aggregated adsr parameters that are longer than signal duration
    adsr_length_in_sec = attack_t + decay_t + sustain_t + release_t

    adsr_clamp_indices = torch.nonzero(adsr_length_in_sec >= synth.SIGNAL_DURATION_SEC, as_tuple=True)[0]

    normalized_attack_list = []
    normalized_decay_list = []
    normalized_sustain_list = []
    normalized_release_list = []

    for i in range(adsr_length_in_sec.shape[0]):
        if i in adsr_clamp_indices.tolist():
            # add small number to normalization to prevent numerical issue where the sum exceeds 1
            normalization_value = adsr_length_in_sec[i] + 1e-3
            normalized_attack = attack_t[i] / normalization_value
            normalized_decay = decay_t[i] / normalization_value
            normalized_sustain = sustain_t[i] / normalization_value
            normalized_release = release_t[i] / normalization_value

        else:
            normalized_attack = attack_t[i]
            normalized_decay = decay_t[i]
            normalized_sustain = sustain_t[i]
            normalized_release = release_t[i]

        normalized_attack_list.append(normalized_attack)
        normalized_decay_list.append(normalized_decay)
        normalized_sustain_list.append(normalized_sustain)
        normalized_release_list.append(normalized_release)

    normalized_attack_tensor = torch.stack(normalized_attack_list)
    normalized_decay_tensor = torch.stack(normalized_decay_list)
    normalized_sustain_tensor = torch.stack(normalized_sustain_list)
    normalized_release_tensor = torch.stack(normalized_release_list)

    clamped_params_dict['attack_t'] = normalized_attack_tensor
    clamped_params_dict['decay_t'] = normalized_decay_tensor
    clamped_params_dict['sustain_t'] = normalized_sustain_tensor
    clamped_params_dict['release_t'] = normalized_release_tensor

    clamped_params_dict['sustain_level'] = torch.clamp(parameters_dict['sustain_level'], min=0, max=synth.MAX_AMP)

    # Add Classification parameters as-is
    clamped_params_dict['osc1_freq'] = parameters_dict['osc1_freq']
    clamped_params_dict['osc2_freq'] = parameters_dict['osc2_freq']
    clamped_params_dict['osc1_wave'] = parameters_dict['osc1_wave']
    clamped_params_dict['osc2_wave'] = parameters_dict['osc2_wave']
    clamped_params_dict['filter_type'] = parameters_dict['filter_type']

    return clamped_params_dict


class Normalizer:
    """ normalize/de-normalise regression parameters"""
    '''
    ['osc1_amp', 'osc1_mod_index', 'lfo1_freq', 'lfo1_phase',
     'osc2_amp', 'osc2_mod_index', 'lfo2_freq', 'lfo2_phase',
     'filter_freq', 'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']'''

    def __init__(self):
        self.mod_index_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=synth.MAX_MOD_INDEX)

        self.lfo_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                    target_max_val=1,
                                                    original_min_val=0,
                                                    original_max_val=synth.MAX_LFO_FREQ)

        self.lfo_phase_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=TWO_PI)

        self.adsr_normalizer = MinMaxNormaliser(target_min_val=0,
                                                target_max_val=1,
                                                original_min_val=0,
                                                original_max_val=synth.SIGNAL_DURATION_SEC)

        self.filter_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                       target_max_val=1,
                                                       original_min_val=0,
                                                       original_max_val=synth.MAX_FILTER_FREQ)

    def normalize(self, parameters_dict: dict):
        normalized_params_dict = {
            'osc1_mod_index': self.mod_index_normalizer.normalise(parameters_dict['osc1_mod_index']),
            'lfo1_freq': self.lfo_freq_normalizer.normalise(parameters_dict['lfo1_freq']),
            'lfo1_phase': self.lfo_phase_normalizer.normalise(parameters_dict['lfo1_phase']),
            'osc2_mod_index': self.mod_index_normalizer.normalise(parameters_dict['osc2_mod_index']),
            'lfo2_freq': self.lfo_freq_normalizer.normalise(parameters_dict['lfo2_freq']),
            'filter_freq': self.filter_freq_normalizer.normalise(parameters_dict['filter_freq']),
            'attack_t': self.adsr_normalizer.normalise(parameters_dict['attack_t']),
            'decay_t': self.adsr_normalizer.normalise(parameters_dict['decay_t']),
            'sustain_t': self.adsr_normalizer.normalise(parameters_dict['sustain_t']),
            'release_t': self.adsr_normalizer.normalise(parameters_dict['release_t'])}

        return normalized_params_dict

    def denormalize(self, parameters_dict: dict):

        if SYNTH_TYPE == 'OSC_ONLY':
            denormalized_params_dict = {'osc1_freq': parameters_dict['osc1_freq']}
        else:
            denormalized_params_dict = {
                'osc1_mod_index': self.mod_index_normalizer.denormalise(parameters_dict['osc1_mod_index']),
                'lfo1_freq': self.lfo_freq_normalizer.denormalise(parameters_dict['lfo1_freq']),
                'osc2_mod_index': self.mod_index_normalizer.denormalise(parameters_dict['osc2_mod_index']),
                'lfo2_freq': self.lfo_freq_normalizer.denormalise(parameters_dict['lfo2_freq']),
                'filter_freq': self.filter_freq_normalizer.denormalise(parameters_dict['filter_freq']),
                'attack_t': self.adsr_normalizer.denormalise(parameters_dict['attack_t']),
                'decay_t': self.adsr_normalizer.denormalise(parameters_dict['decay_t']),
                'sustain_t': self.adsr_normalizer.denormalise(parameters_dict['sustain_t']),
                'release_t': self.adsr_normalizer.denormalise(parameters_dict['release_t']),

                # params that doesn't need denormalization:
                'osc1_freq': parameters_dict['osc1_freq'],
                'osc1_wave': parameters_dict['osc1_wave'],
                'osc1_amp': parameters_dict['osc1_amp'],
                'osc2_freq': parameters_dict['osc2_freq'],
                'osc2_wave': parameters_dict['osc2_wave'],
                'osc2_amp': parameters_dict['osc2_amp'],
                'filter_type': parameters_dict['filter_type'],
                'sustain_level': parameters_dict['sustain_level']}

        return denormalized_params_dict


def plot_spectrogram(spec, scale='linear', title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    if scale == 'linear':
        im = axs.imshow(spec, origin='lower', aspect=aspect)
    elif scale == 'dB:':
        im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to a tensor"""

    def __init__(self, target_min_val, target_max_val, original_min_val, original_max_val):
        self.target_min_val = target_min_val
        self.target_max_val = target_max_val
        self.original_max_val = original_min_val
        self.original_min_val = original_max_val

    def normalise(self, array):
        norm_array = (array - self.original_min_val) / (self.original_max_val - self.original_min_val)
        return norm_array

    def denormalise(self, norm_array):
        array = (norm_array - self.target_min_val) / (self.target_max_val - self.target_min_val)
        array = array * (self.original_max_val - self.original_min_val) + self.original_min_val
        return array


class LogNormaliser:
    """LogNormaliser applies log normalisation to a tensor"""

    def __init__(self):
        pass

    def normalise(self, array):
        # add small value to prevent -inf for log(0)
        norm_array = torch.log(array + 1e-10)
        return norm_array

    def denormalise(self, norm_array):
        array = torch.exp(norm_array) - 1e-10
        return array

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


# from https://github.com/pytorch/pytorch/issues/61292
def linspace(start: Tensor, stop: Tensor, num: Tensor):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    start = start.to(get_device())
    stop = stop.to(get_device())
    num = num.to(get_device())
    # create a tensor of 'num' steps from 0 to 1

    # todo: make sure ADSR behavior is differentiable. arange has to know to get tensors
    # OPTION1
    # arange_list_of_tensors = []
    # current_tensor = torch.tensor([0], dtype=torch.float32, device=get_device(), requires_grad=True)
    # while current_tensor.item() < num.item():
    #     arange_list_of_tensors.append(current_tensor)
    #     current_tensor = current_tensor + 1
    #
    # arange_tensor1 = torch.stack(arange_list_of_tensors).to(get_device()).squeeze()

    # OPTION2
    arange_tensor2 = torch.arange(num, dtype=torch.float32, device=get_device(), requires_grad=True)

    steps = arange_tensor2 / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out
